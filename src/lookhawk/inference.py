"""Runtime integration: swap HAWK's score source for a learned scorer.

The patched ``Qwen2_5_VLModel.forward`` executes ``from hawk.pruning import
prune_qwen_prefill`` *inside the method body on every prefill*, so swapping the
module attribute ``hawk.pruning.prune_qwen_prefill`` before generation cleanly
replaces the score source without touching any file in ``src/hawk/`` or the
Transformers runtime. Bookkeeping (``_store_state``, ``PrunedPrefill``,
``select_keep_indices``, PruningStats) is reused verbatim, so the decode-side
patch and the VLMEvalKit adapters behave exactly as they do under HAWK.

``install_lookahead(model, scorer)`` stores the scorer on the inner model ``W``
and rebinds the HAWK function. When no scorer is installed, or the sample has no
visual tokens / pruning is disabled, the original HAWK function handles the
call (identical no-op bookkeeping).

Two scorer designs are supported and the *type* of the installed scorer picks
between them -- there is no separately stored mode to go stale:

* :class:`~lookhawk.modules.Layer0Scorer` -- appended rows are read through
  decoder layer 0 before scoring (``read.py``).
* :class:`~lookhawk.layer1.PseudoTokenScorer` -- the full forward runs with real
  mRoPE and decoder layer 1's own ``q_proj``/``k_proj`` score the read-out
  hidden state **pre-RoPE** (``layer1.py``). This needs the first two decoder
  layers to exist, so it temporarily trims ``text.layers`` for the scoring pass
  only (see :func:`_layer1_visual_scores`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

import hawk.pruning
from hawk.config import HawkConfig
from hawk.pruning import (
    PrunedPrefill,
    PruningStats,
    _store_state,
    compute_hawk_scores,
    select_keep_indices,
)

from .layer1 import PseudoTokenScorer, extend_position_ids, pre_rope_read, visual_scores
from .model_graph import (
    IMAGE_TOKEN_ID,
    VIDEO_TOKEN_ID,
    ModelHandle,
    resolve_handle,
)
from .modules import Layer0Scorer
from .read import conditioned_layer0_read, score_from_conditioned

# Captured at import, before anything can rebind it. Reading
# ``hawk.pruning.prune_qwen_prefill`` at install time instead would hand a second
# installed model the lookahead function as its "original", and its fallback
# would recurse into itself.
_ORIGINAL_PRUNE = hawk.pruning.prune_qwen_prefill


class LookhawkError(RuntimeError):
    pass


def _image_token_ids(handle: ModelHandle) -> tuple[int, int]:
    config = handle.config()
    return (
        int(getattr(config, "image_token_id", IMAGE_TOKEN_ID)),
        int(getattr(config, "video_token_id", VIDEO_TOKEN_ID)),
    )


def _align_scorer(scorer: nn.Module, handle: ModelHandle) -> nn.Module:
    """Move/cast the scorer to the decoder weights' device and dtype."""
    parameter = next(handle.layer0().parameters())
    return scorer.to(device=parameter.device, dtype=parameter.dtype)


def install_lookahead(
    model: Any,
    scorer: nn.Module,
    cfg: HawkConfig | None = None,
) -> Any:
    """Enable lookahead scoring on a loaded Qwen2.5-VL model.

    ``model`` may be the outer conditional-generation model or the inner
    ``W``. The scorer is stored on ``W`` (the state-owner), aligned to the
    decoder weight dtype/device, and ``hawk.pruning.prune_qwen_prefill`` is
    rebound to :func:`prune_qwen_prefill_lookahead`. Whether that function takes
    the layer-0 or the layer-1 branch is decided by ``isinstance(scorer,
    PseudoTokenScorer)``.
    """
    handle = resolve_handle(model)
    wrapper = handle.wrapper
    aligned = _align_scorer(scorer, handle)
    aligned.eval()

    wrapper._lookhawk_scorer = aligned
    wrapper._lookhawk_cfg = cfg

    if not getattr(wrapper, "_lookhawk_orig_prune", None):
        wrapper._lookhawk_orig_prune = _ORIGINAL_PRUNE
    if hawk.pruning.prune_qwen_prefill is not prune_qwen_prefill_lookahead:
        hawk.pruning.prune_qwen_prefill = prune_qwen_prefill_lookahead
    return model


def uninstall_lookahead(model: Any) -> None:
    handle = resolve_handle(model)
    wrapper = handle.wrapper
    if getattr(wrapper, "_lookhawk_orig_prune", None) is not None:
        wrapper._lookhawk_orig_prune = None
    wrapper._lookhawk_scorer = None
    if hawk.pruning.prune_qwen_prefill is prune_qwen_prefill_lookahead:
        hawk.pruning.prune_qwen_prefill = _ORIGINAL_PRUNE


@torch.no_grad()
def prune_qwen_prefill_lookahead(
    model: Any,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    cache_position: torch.Tensor | None,
    config: HawkConfig,
) -> PrunedPrefill:
    """Drop-in replacement for :func:`hawk.pruning.prune_qwen_prefill`.

    Signature and return type match HAWK exactly. The appended lookahead rows
    are used only to *score* visual tokens; they never appear in the returned
    ``inputs_embeds``, so cache_position/rope_deltas bookkeeping is unaffected.
    """
    orig = getattr(model, "_lookhawk_orig_prune", None)
    scorer = getattr(model, "_lookhawk_scorer", None)
    handle = resolve_handle(model)
    image_token_id, video_token_id = _image_token_ids(handle)

    row = input_ids[0]
    visual_mask = (row == image_token_id) | (row == video_token_id)
    if attention_mask is not None:
        visual_mask = visual_mask & attention_mask[0].bool()
    visual_indices = torch.where(visual_mask)[0].long()
    sequence_length = int(inputs_embeds.shape[1])

    # Non-lookahead / no-op conditions fall back to the HAWK function so the
    # bookkeeping (stats object, applied flag) is byte-for-byte identical.
    if (
        orig is None
        or scorer is None
        or config.keep_ratio >= 1.0
        or visual_indices.numel() == 0
    ):
        return orig(
            model=model,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            config=config,
        )

    if isinstance(scorer, PseudoTokenScorer):
        scores, query_tokens = _layer1_visual_scores(
            handle, scorer, inputs_embeds, visual_indices, position_ids
        )
        normalization = f"softmax_visual/{scorer.head_merge}"
    else:
        scores, query_tokens = _layer0_visual_scores(
            handle, scorer, inputs_embeds, visual_indices, config
        )
        normalization = config.score_normalization

    return _prune_result(
        model=model,
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        config=config,
        visual_indices=visual_indices,
        scores=scores,
        query_tokens=query_tokens,
        sequence_length=sequence_length,
        # Reported in the stats/report. The layer-1 score is a probability
        # distribution over visual columns, not an L2-normalised score, so
        # echoing ``config.score_normalization`` there would be a lie.
        normalization=normalization,
    )


def _layer0_visual_scores(
    handle: ModelHandle,
    scorer: Layer0Scorer,
    inputs_embeds: torch.Tensor,
    visual_indices: torch.Tensor,
    config: HawkConfig,
) -> tuple[torch.Tensor, int]:
    """Conditioned layer-0 read of the appended rows -> ``[V]`` scores."""
    rows = scorer.embed_lookahead.unsqueeze(0).to(
        dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    seq = torch.cat([inputs_embeds, rows], dim=1)
    appended_hidden = conditioned_layer0_read(
        handle.layer0(), seq, appended_size=int(rows.shape[1]), scorer=scorer
    )
    head_weights = torch.as_tensor(
        list(config.head_weights), device=inputs_embeds.device, dtype=torch.float32
    )
    scores = score_from_conditioned(
        handle.layer0(),
        inputs_embeds,
        visual_indices,
        appended_hidden,
        head_weights,
        normalization=config.score_normalization,
        scorer=scorer,
    )
    return scores, int(rows.shape[1])


def _layer1_visual_scores(
    handle: ModelHandle,
    scorer: PseudoTokenScorer,
    inputs_embeds: torch.Tensor,
    visual_indices: torch.Tensor,
    position_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    """Score visual tokens from decoder layer ``scorer.score_layer`` -> ``[V]``.

    Mirrors what :class:`~lookhawk.layer1.Layer1Trainer` does at training time:
    append the learned rows, run the real (mRoPE) forward, then dot layer
    ``score_layer``'s ``q``/``k`` **pre-RoPE** and softmax over the visual
    columns.
    """
    if position_ids is None or position_ids.ndim != 3:
        # generate() always reaches the prefill branch with [3, B, T] (it sets
        # position_ids to None and Qwen2_5_VLModel rebuilds it from
        # get_rope_index). Anything else is a hand-rolled call the design cannot
        # score -- say so rather than silently falling back to HAWK's scores.
        raise LookhawkError(
            "layer-1 lookahead scoring needs [3, B, T] mRoPE position_ids during "
            f"prefill; got {None if position_ids is None else tuple(position_ids.shape)}"
        )

    text = handle.text
    layers = text.layers
    if scorer.score_layer >= len(layers):
        raise LookhawkError(
            f"score_layer={scorer.score_layer} but the text decoder has {len(layers)} layers"
        )

    rows = scorer.embed_pseudo.unsqueeze(0).to(
        dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    num_rows = int(rows.shape[1])
    positions = extend_position_ids(position_ids, num_rows)

    # Only the first ``score_layer + 1`` layers are needed. Trimming is purely a
    # cost optimisation, not a correctness device: ``hidden_states[score_layer]``
    # is the causal output of the layers below it, so it is bit-identical to the
    # same entry in a 28-layer forward. Restore the *same* ModuleList object.
    try:
        text.layers = nn.ModuleList(list(layers[: scorer.score_layer + 1]))
        attention = pre_rope_read(
            text, inputs_embeds, rows, positions, visual_indices, scorer.score_layer
        )
    finally:
        text.layers = layers

    scores = visual_scores(
        attention, temperature=scorer.temperature, head_merge=scorer.head_merge
    )
    return scores, num_rows


def _prune_result(
    *,
    model: Any,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    cache_position: torch.Tensor | None,
    config: HawkConfig,
    visual_indices: torch.Tensor,
    scores: torch.Tensor,
    query_tokens: int,
    sequence_length: int,
    normalization: str | None,
) -> PrunedPrefill:
    """Shared tail: select, build stats, store generation state, prune.

    Kept in one place so both scorer designs produce byte-for-byte identical
    bookkeeping -- which is what makes the decode-side patch and the VLMEvalKit
    adapters behave exactly as they do under HAWK.
    """
    keep_indices, selected_visual = select_keep_indices(
        visual_indices,
        scores,
        sequence_length,
        config.keep_ratio,
        config.min_visual_tokens,
    )

    kept_visual = int(selected_visual.numel())
    original_visual = int(visual_indices.numel())
    kept_total = int(keep_indices.numel())
    stats = PruningStats(
        applied=kept_visual < original_visual,
        original_total_tokens=sequence_length,
        kept_total_tokens=kept_total,
        original_visual_tokens=original_visual,
        kept_visual_tokens=kept_visual,
        removed_visual_tokens=original_visual - kept_visual,
        keep_ratio_requested=config.keep_ratio,
        keep_ratio_actual=kept_visual / original_visual,
        query_tokens=query_tokens,
        score_normalization=normalization,
    )
    pruned_attention_mask = (
        attention_mask.index_select(1, keep_indices.to(attention_mask.device))
        if attention_mask is not None
        else None
    )
    pruned_position_ids = (
        position_ids.index_select(2, keep_indices.to(position_ids.device))
        if position_ids is not None
        else None
    )
    pruned_cache_position = (
        cache_position[:kept_total] if cache_position is not None else None
    )
    _store_state(model, keep_indices, stats)

    return PrunedPrefill(
        inputs_embeds=inputs_embeds.index_select(1, keep_indices),
        attention_mask=pruned_attention_mask,
        position_ids=pruned_position_ids,
        cache_position=pruned_cache_position,
        keep_indices=keep_indices,
        stats=stats,
    )


def load_lookahead_for_model(
    model: Any,
    checkpoint_path: str | Path,
    hawk_config: HawkConfig | None = None,
) -> nn.Module:
    """Load a scorer checkpoint and install it on an already-configured model.

    The checkpoint's ``kind`` selects the design: ``"pseudo_token"`` builds the
    layer-1 scorer (which carries its own ``score_layer``/``temperature``/
    ``head_merge``), anything else falls back to the layer-0 scorer.
    """
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    handle = resolve_handle(model)
    if payload.get("kind") == "pseudo_token":
        from .layer1 import load_scorer

        scorer: nn.Module = load_scorer(str(checkpoint_path))
        if int(scorer.hidden_size) != int(handle.text.embed_tokens.weight.shape[1]):
            raise LookhawkError(
                f"checkpoint hidden size {scorer.hidden_size} does not match the model's "
                f"{handle.text.embed_tokens.weight.shape[1]}"
            )
    else:
        from .train import load_scorer_from_layer0

        scorer = load_scorer_from_layer0(handle.layer0(), checkpoint_path)
    install_lookahead(model, scorer, cfg=hawk_config)
    return scorer
