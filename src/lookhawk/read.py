"""Layer-0 conditioning read and HAWK-compatible visual scoring.

The unifying primitive: append a block of ``appended_size`` rows after the
prompt embeddings, run them through ONE decoder layer-0 block (no RoPE, causal,
querying only the appended rows so prompt rows can never see them), then turn
the conditioned appended hidden states into HAWK visual-token scores with the
exact same aggregation the training-free HAWK path uses.

The same code path serves three roles:

* teacher -- appended rows are real answer-token embeddings, ``scorer=None``;
* student / inference -- appended rows are the learned ``embed_lookahead`` and
  ``scorer`` applies the additive LoRA to those rows during the read.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from hawk.pruning import compute_hawk_scores, repeat_kv

from .model_graph import IMAGE_TOKEN_ID, ModelHandle


def _projection_out_features(projection: Any) -> int:
    if hasattr(projection, "out_features"):
        return int(projection.out_features)
    return int(projection.in_features)


def _visual_tokens(ids: torch.Tensor, image_token_id: int) -> int:
    return int((ids == image_token_id).sum().item())


@torch.no_grad()
def build_prompt_embeds(
    handle: ModelHandle,
    prompt_ids: torch.Tensor,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
) -> torch.Tensor:
    """Embed a prompt and scatter vision features, mirroring the Qwen forward.

    ``prompt_ids``: ``[1, T]`` on the target device. Returns ``[1, T, hidden]``
    with vision features placed at the ``image_token_id`` positions. Requires
    ``handle.visual`` only when ``pixel_values`` is provided.
    """
    if prompt_ids.ndim != 2:
        raise ValueError("prompt_ids must be rank-2")
    config = handle.config()
    image_token_id = int(getattr(config, "image_token_id", IMAGE_TOKEN_ID))
    embeds = handle.embed_tokens()(prompt_ids)
    if pixel_values is None:
        return embeds
    if handle.visual is None:
        raise TypeError("pixel_values given but no vision tower resolved")

    visual_dtype = next(handle.visual.parameters()).dtype
    image_embeds = handle.visual(pixel_values.to(visual_dtype), grid_thw=image_grid_thw)
    expected = _visual_tokens(prompt_ids, image_token_id)
    if image_embeds.shape[0] != expected:
        raise RuntimeError(
            f"vision tower produced {image_embeds.shape[0]} features but the prompt "
            f"has {expected} image tokens"
        )
    image_embeds = image_embeds.to(embeds.dtype)
    mask = (prompt_ids == image_token_id).unsqueeze(-1).expand_as(embeds)
    return embeds.masked_scatter(mask, image_embeds)


def _attn_shapes(layer0: Any) -> tuple[int, int, int, int, int, int]:
    attention = layer0.self_attn
    return (
        int(attention.num_heads),
        int(attention.num_key_value_heads),
        int(attention.num_key_value_groups),
        int(attention.head_dim),
        int(attention.q_proj.in_features),
        _projection_out_features(attention.q_proj),
    )


def _attention_projection_states(layer0: Any) -> tuple[Any, Any, Any, Any]:
    attention = layer0.self_attn
    return attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj


def conditioned_layer0_read(
    layer0: Any,
    seq_embeds: torch.Tensor,
    appended_size: int,
    scorer: Any | None = None,
) -> torch.Tensor:
    """Run one no-RoPE causal layer-0 block for the appended rows.

    ``seq_embeds``: ``[1, T + appended_size, hidden]``; the last
    ``appended_size`` rows are the appended block and may attend, causally, the
    entire prompt plus earlier appended rows. Returns the ``[1, appended_size,
    hidden]`` block outputs. If ``scorer`` is provided its additive LoRAs are
    applied to the appended rows' q/k (and v/o / MLP when enabled).
    """
    if seq_embeds.ndim != 3:
        raise ValueError("seq_embeds must be rank-3 [1, seq, hidden]")
    length = seq_embeds.shape[1]
    if appended_size < 1 or appended_size > length:
        raise ValueError(f"appended_size must be in [1, {length}]")
    prefix_len = length - appended_size
    attention = layer0.self_attn
    (
        num_heads,
        num_kv_heads,
        groups,
        head_dim,
        hidden_size,
        q_out,
    ) = _attn_shapes(layer0)
    q_proj, k_proj, v_proj, o_proj = _attention_projection_states(layer0)
    if q_out != hidden_size:
        raise ValueError(
            f"expected q projection out_features==hidden_size ({hidden_size}), "
            f"got {q_out}"
        )
    if int(k_proj.out_features) != num_kv_heads * head_dim:
        raise ValueError("k projection shape mismatch with num_key_value_heads*head_dim")

    normalized = layer0.input_layernorm(seq_embeds)
    appended_normed = normalized[:, -appended_size:, :]

    # Keys and values cover the full causal prefix.
    key_states = k_proj(normalized)
    value_states = v_proj(normalized)
    k_lora = scorer.proj_lora("k") if scorer is not None else None
    v_lora = scorer.proj_lora("v") if scorer is not None else None
    if k_lora is not None:
        key_states = key_states.clone()
        key_states[:, -appended_size:, :] += k_lora(appended_normed)
    if v_lora is not None:
        value_states = value_states.clone()
        value_states[:, -appended_size:, :] += v_lora(appended_normed)

    batch, full_len, _ = key_states.shape
    key_states = key_states.view(batch, full_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch, full_len, num_kv_heads, head_dim).transpose(1, 2)
    key_states = repeat_kv(key_states, groups)
    value_states = repeat_kv(value_states, groups)

    # Queries only for the appended rows.
    query_states = q_proj(appended_normed)
    q_lora = scorer.proj_lora("q") if scorer is not None else None
    if q_lora is not None:
        query_states = query_states + q_lora(appended_normed)
    _, q_len, _ = query_states.shape
    query_states = query_states.view(batch, q_len, num_heads, head_dim).transpose(1, 2)

    # Windowed causal mask: appended row i may attend keys 0 .. prefix_len + i.
    row = torch.arange(appended_size, device=seq_embeds.device).view(-1, 1)
    col = torch.arange(full_len, device=seq_embeds.device).view(1, -1)
    causal_window = col <= (prefix_len + row)

    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, attn_mask=causal_window
    )
    attn_output = attn_output.transpose(1, 2).reshape(batch, q_len, hidden_size)
    out_proj = o_proj(attn_output)
    o_lora = scorer.proj_lora("o") if scorer is not None else None
    if o_lora is not None:
        out_proj = out_proj + o_lora(attn_output)

    residual = seq_embeds[:, -appended_size:, :]
    mid = residual + out_proj
    mlp_input = layer0.post_attention_layernorm(mid)
    mlp_output = _mlp_forward(layer0, mlp_input, scorer)
    return mid + mlp_output


def _mlp_forward(layer0: Any, x: torch.Tensor, scorer: Any | None) -> torch.Tensor:
    """MLP forward with optional per-projection additive LoRA on the rows."""
    mlp = layer0.mlp
    gate = mlp.gate_proj(x)
    up = mlp.up_proj(x)
    gate_lora = scorer.proj_lora("gate") if scorer is not None else None
    up_lora = scorer.proj_lora("up") if scorer is not None else None
    down_lora = scorer.proj_lora("down") if scorer is not None else None
    if gate_lora is not None:
        gate = gate + gate_lora(x)
    if up_lora is not None:
        up = up + up_lora(x)
    act_fn = getattr(mlp, "act_fn", None)
    if act_fn is None:
        raise TypeError("decoder layer MLP has no act_fn")
    hidden = act_fn(gate) * up
    down = mlp.down_proj(hidden)
    if down_lora is not None:
        down = down + down_lora(hidden)
    return down


def score_from_conditioned(
    layer0: Any,
    prompt_embeds: torch.Tensor,
    visual_indices: torch.Tensor,
    appended_hidden: torch.Tensor,
    head_weights: torch.Tensor,
    normalization: str = "l2",
    scorer: Any | None = None,
) -> torch.Tensor:
    """Score visual tokens from conditioned appended hidden rows.

    ``appended_hidden`` is the ``[1, L, hidden]`` output of
    :func:`conditioned_layer0_read`. Queries use the layer-0 ``q_proj`` after
    ``input_layernorm``; when ``scorer`` is given its score-stage ``q`` LoRA is
    added to the frozen query (same LN space, additive), so the learned
    parameters genuinely steer the query while the read-conditioned input keeps
    it sample-dependent. Keys are the pre-RoPE ``k_proj`` of the prompt's visual
    tokens, and aggregation goes through HAWK's ``compute_hawk_scores`` so
    training matches inference exactly.
    """
    if appended_hidden.ndim != 3:
        raise ValueError("appended_hidden must be rank-3")
    num_heads, num_kv_heads, groups, head_dim, _, _ = _attn_shapes(layer0)
    q_proj, k_proj, _, _ = _attention_projection_states(layer0)
    device = appended_hidden.device

    normalized_appended = layer0.input_layernorm(appended_hidden)
    query_states = q_proj(normalized_appended)
    if scorer is not None:
        score_q_lora = scorer.score_lora("q")
        if score_q_lora is not None:
            query_states = query_states + score_q_lora(normalized_appended)
    batch, q_len, _ = query_states.shape
    query_states = query_states.view(batch, q_len, num_heads, head_dim).transpose(1, 2)

    normalized_prompt = layer0.input_layernorm(prompt_embeds)
    key_states = k_proj(normalized_prompt)
    kv_len = key_states.shape[1]
    key_states = key_states.view(batch, kv_len, num_kv_heads, head_dim).transpose(1, 2)
    key_states = repeat_kv(key_states, groups)
    visual_keys = key_states.index_select(2, visual_indices.to(device))

    weights = head_weights.to(device=device, dtype=torch.float32)
    return compute_hawk_scores(
        query_states,
        visual_keys,
        weights,
        normalization=normalization,
    )


def score_from_query_embeds(
    layer0: Any,
    prompt_embeds: torch.Tensor,
    visual_indices: torch.Tensor,
    query_embeds: torch.Tensor,
    head_weights: torch.Tensor,
    normalization: str = "l2",
) -> torch.Tensor:
    """Score visual tokens from arbitrary query-token rows, without a read.

    ``query_embeds`` ``[1, N, hidden]`` are token embeddings (real answer rows
    for the genuine teacher, or instruction-text rows for the HAWK baseline).
    Their pre-attention ``q = q_proj(layer0.input_layernorm(query_embeds))`` is
    aggregated against the prompt's pre-RoPE visual keys with HAWK's exact
    ``compute_hawk_scores`` transform. This is the *genuine* target: no
    conditioning read, so the query content (what the future token would look
    like) is not washed out.
    """
    if query_embeds.ndim != 3:
        raise ValueError("query_embeds must be rank-3 [1, N, hidden]")
    num_heads, num_kv_heads, groups, head_dim, _, _ = _attn_shapes(layer0)
    q_proj, k_proj, _, _ = _attention_projection_states(layer0)
    device = query_embeds.device

    query_states = q_proj(layer0.input_layernorm(query_embeds))
    batch, q_len, _ = query_states.shape
    query_states = query_states.view(batch, q_len, num_heads, head_dim).transpose(1, 2)

    normalized_prompt = layer0.input_layernorm(prompt_embeds)
    key_states = k_proj(normalized_prompt)
    kv_len = key_states.shape[1]
    key_states = key_states.view(batch, kv_len, num_kv_heads, head_dim).transpose(1, 2)
    key_states = repeat_kv(key_states, groups)
    visual_keys = key_states.index_select(2, visual_indices.to(device))

    weights = head_weights.to(device=device, dtype=torch.float32)
    return compute_hawk_scores(
        query_states,
        visual_keys,
        weights,
        normalization=normalization,
    )


def teacher_rows_from_answer(
    answer_ids: torch.Tensor,
    max_rows: int,
    selection: str = "first",
) -> torch.Tensor:
    """Select the answer-token rows that form the teacher conditioning block.

    ``answer_ids``: ``[K]`` (already stripped of trailing end-turn tokens).
    Returns the row indices into the answer, at most ``max_rows``, in original
    order. ``selection="first"`` keeps the leading rows; ``"strided"`` spreads
    across a long answer so dependence on later tokens is also supervised.
    """
    if answer_ids.ndim != 1:
        raise ValueError("answer_ids must be rank-1")
    count = answer_ids.numel()
    if count == 0:
        raise ValueError("answer_ids is empty")
    if selection == "first":
        return torch.arange(min(max_rows, count), device=answer_ids.device)
    if selection == "strided":
        step = (count + max_rows - 1) // max_rows
        rows = torch.arange(0, count, step, device=answer_ids.device)
        if rows[-1] != count - 1:
            rows = torch.cat([rows, torch.tensor([count - 1], device=rows.device)])
        return rows[:max_rows]
    raise ValueError(f"unknown teacher_selection {selection!r}")


def select_teacher_answer_ids(
    answer_ids: torch.Tensor,
    cfg: Any,
) -> torch.Tensor:
    """Return the answer token ids used as the teacher's appended rows."""
    return answer_ids[teacher_rows_from_answer(answer_ids, cfg.teacher_max_rows, cfg.teacher_selection)]
