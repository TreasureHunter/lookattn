"""Layer-1 pre-RoPE read: pseudo tokens scored by layer 1's own projections.

This replaces the layer-0 HAWK-aggregation path (``read.py`` + ``modules.py``)
with HAWK's scoring recipe applied one layer deeper.

The idea, per the design discussion:

* Put ``P`` learned pseudo rows at the end of the prompt and run the model
  normally. Through layers 0 and 1 they read the whole prompt exactly as any
  other token does -- no hand-written attention, no special-casing.
* The forward runs with real mRoPE, so the hidden state layer 1 reads is the
  genuine one. But the **score** is not layer 1's softmax attention: it is
  ``q_proj``/``k_proj`` of layer 1 applied to that hidden state and dotted
  **without RoPE** -- the same quantity HAWK takes from layer 0, one layer
  deeper. Queries are the appended rows, keys the visual positions.
* The teacher is the same thing for the **real answer tokens**: append those
  instead and read their scores over the same visual positions.
* Loss: per-head KL(student || teacher) over the visual tokens, averaged over
  the query rows first and summed over heads (the LookaheadKV shape).

Why pre-RoPE -- the deliberate trade taken here:

* RoPE makes the score depend on the *relative distance* between an appended
  row and each visual token, and that distance scales with the visual-token
  count ``V``. Qwen's mRoPE additionally gives every visual key a ``(t, h, w)``
  address, so the score becomes position-structured. A scorer trained at one
  ``V`` then means something different at another, and ``V`` ranges over
  1280..16384 during evaluation.
* Dropping RoPE leaves a pure content match, independent of where a token sits
  -- the property HAWK's training-free scores rely on. The cost is real: the
  ``(h, w)`` spatial address the model would otherwise use is thrown away, and
  the target is now a *pre-RoPE proxy* for where the answer looks rather than
  the model's real attention.
* Two side benefits: the forward no longer needs ``output_attentions=True`` (so
  it runs on sdpa instead of falling back to eager), and the ``[H, T, T]``
  attention matrix is never materialised.

One more problem the layered-0 design had, still fixed: no LoRA, and per-head
supervision with no static ``head_weights`` to collapse heads.

Cost: two full-prompt forward passes per step instead of one layer-0 block. Both
only run decoder layers 0..score_layer, so load the model trimmed
(``load_and_trim_model(..., keep_layers=score_layer + 1)``).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from hawk.pruning import repeat_kv


# ------------------------------------------------------------------ module

class PseudoTokenScorer(nn.Module):
    """The whole trainable state: ``P`` rows appended to the prompt embeddings.

    No LoRA. The parameter count is ``P * hidden`` (32 * 3584 ~ 115k floats).
    """

    def __init__(
        self,
        hidden_size: int,
        num_pseudo: int = 32,
        init_scale: float = 0.02,
        score_layer: int = 1,
        temperature: float = 1.0,
        head_merge: str = "mean",
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_pseudo = int(num_pseudo)
        # How to *read* the score out of this scorer. Kept on the module (not in
        # the surrounding ``HawkConfig``, which has no such fields) so a
        # checkpoint carries the settings it was trained with: a mismatch here
        # -- temperature 0.5 at training, 1.0 at inference -- would silently
        # rank the visual tokens differently instead of failing.
        self.score_layer = int(score_layer)
        self.temperature = float(temperature)
        self.head_merge = str(head_merge)
        self.embed_pseudo = nn.Parameter(
            torch.randn(self.num_pseudo, self.hidden_size) * init_scale
        )

    @classmethod
    def from_handle(
        cls,
        handle: Any,
        num_pseudo: int = 32,
        init_scale: float = 0.02,
        score_layer: int = 1,
        temperature: float = 1.0,
        head_merge: str = "mean",
    ) -> "PseudoTokenScorer":
        hidden = int(handle.layer0().self_attn.q_proj.in_features)
        return cls(
            hidden,
            num_pseudo=num_pseudo,
            init_scale=init_scale,
            score_layer=score_layer,
            temperature=temperature,
            head_merge=head_merge,
        )


# ------------------------------------------------------------ position ids

def extend_position_ids(position_ids: torch.Tensor, extra: int) -> torch.Tensor:
    """Append ``extra`` positions continuing from the last one.

    ``position_ids`` is Qwen2.5-VL's 3-row mRoPE layout ``[3, B, T]``. The
    appended rows are ordinary text tokens sitting right after the prompt, so
    each of the three rows simply continues counting -- the same positions the
    real answer tokens would occupy.
    """
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError(f"expected [3, B, T] mRoPE position_ids, got {tuple(position_ids.shape)}")
    last = position_ids[:, :, -1:]
    steps = torch.arange(1, extra + 1, device=position_ids.device).view(1, 1, -1)
    return torch.cat([position_ids, last + steps], dim=2)


# ------------------------------------------------------------ attention read

def pre_rope_read(
    text_decoder: Any,
    prompt_embeds: torch.Tensor,
    appended_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    visual_indices: torch.Tensor,
    layer_index: int,
) -> torch.Tensor:
    """Scores of the appended rows over the visual tokens at ``layer_index``.

    Returns ``[H, Q, V]`` probabilities: ``q_proj``/``k_proj`` of decoder layer
    ``layer_index`` applied to that layer's own input, dotted **without RoPE**
    (HAWK's recipe), then softmaxed over the visual positions. Gradient flows
    back into ``appended_embeds`` (and thus into whatever produced it);
    everything in the model is frozen.

    The forward itself *does* use real mRoPE -- only the scoring projections skip
    it -- so ``hidden_states[layer_index]`` is the genuine hidden state and not
    something produced by an ablated model. ``position_ids`` must already cover
    the *concatenated* sequence, i.e. :func:`extend_position_ids` of the prompt's
    rope index.

    Causality needs no mask here: every visual token precedes every appended row,
    so restricting the softmax to the visual columns is already causal.

    Note the softmax is over the visual columns *only*, whereas the model's own
    attention would normalise over all keys. That is what makes the result a
    distribution over visual tokens, which is the quantity the KL is defined on.
    """
    if prompt_embeds.shape[0] != 1 or appended_embeds.shape[0] != 1:
        raise ValueError("this path is batch-size 1 (variable-length samples)")
    prefix_len = int(prompt_embeds.shape[1])
    query_len = int(appended_embeds.shape[1])
    seq = torch.cat([prompt_embeds, appended_embeds], dim=1)
    total = prefix_len + query_len
    if position_ids.shape[2] != total:
        raise ValueError(
            f"position_ids covers {position_ids.shape[2]} positions but the "
            f"sequence has {total}"
        )
    attention_mask = torch.ones(1, total, dtype=torch.long, device=seq.device)

    outputs = text_decoder(
        inputs_embeds=seq,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        output_hidden_states=True,
        # Spelled out explicitly: inheriting ``config.output_attentions=True``
        # would make ``_update_causal_mask`` skip the SDPA shortcut, fall back to
        # eager, and materialise ``[heads, T+P, T+P]`` weights -- gigabytes at the
        # long-prompt setting this design is meant to run at.
        output_attentions=False,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None or layer_index >= len(hidden_states):
        raise RuntimeError(f"layer {layer_index} hidden state unavailable")
    # ``hidden_states[i]`` is the *input* to decoder layer ``i``.
    hidden = hidden_states[layer_index]

    layer = text_decoder.layers[layer_index]
    attention = layer.self_attn
    normalized = layer.input_layernorm(hidden)

    batch_size, length, _ = normalized.shape
    head_dim = int(attention.head_dim)
    queries = attention.q_proj(normalized).view(batch_size, length, -1, head_dim).transpose(1, 2)
    keys = attention.k_proj(normalized).view(batch_size, length, -1, head_dim).transpose(1, 2)
    keys = repeat_kv(keys, attention.num_key_value_groups)

    row_ids = torch.arange(prefix_len, total, device=seq.device)
    query_states = queries.index_select(2, row_ids)
    key_states = keys.index_select(2, visual_indices.to(seq.device))

    # Pre-RoPE -- the one deliberate difference from the model's own attention.
    logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    return F.softmax(logits.float(), dim=-1)[0]  # [H, Q, V]


# -------------------------------------------------------------------- loss

def _row_averaged_distribution(
    attention: torch.Tensor, temperature: float = 1.0, eps: float = 1e-8
) -> torch.Tensor:
    """``[H, Q, V]`` attention probabilities -> ``[H, V]`` distribution.

    Averages over the query rows *as probabilities*, then re-normalises -- which
    is the LookaheadKV order. Averaging log-probabilities instead would leave a
    tensor that is not a distribution at all, and ``kl_div`` would silently
    return a number that is not a KL.
    """
    probs = attention.float().clamp_min(eps)
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    probs = probs.mean(dim=1)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def _head_weights(teacher: torch.Tensor, mode: str) -> torch.Tensor:
    """Per-head weights, normalised so their mean is 1 (keeps the loss scale)."""
    mass = teacher.float().sum(dim=-1).mean(dim=1)  # [H] mean visual mass per head
    if mode == "visual_mass":
        weight = mass / mass.sum().clamp_min(1e-9)
    elif mode == "student_mass":
        raise ValueError("student_mass needs the student tensor; pass explicit weights")
    else:
        raise ValueError(f"unknown head weight mode {mode!r}")
    return weight * mass.numel()  # mean 1, so the loss stays comparable to a plain sum


def layer1_kl(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float = 1.0,
    head_weights: torch.Tensor | str | None = None,
) -> torch.Tensor:
    """Per-head KL over visual tokens, averaged over query rows, merged over heads.

    ``student`` / ``teacher`` are ``[H, Q, V]`` probabilities (the raw output of
    :func:`pre_rope_read`). Returns ``sum_h w_h KL(t_h || s_h)``
    -- the teacher is the target, so the loss pulls the student onto it.

    The two may have **different query counts**: the row dimension is averaged
    away by :func:`_row_averaged_distribution` before the KL, so ``P`` pseudo
    rows are compared against ``A`` answer rows through their respective means.
    Only the head and visual dims have to line up.

    ``head_weights``:

    * ``None`` -- every head counts equally (the default; start here).
    * ``"visual_mass"`` -- weight each head by the teacher's own visual attention
      mass, so heads that do not look at vision contribute nothing. No free
      parameters: the weights are measured from the teacher, not learned.
    * a ``[H]`` tensor -- used as given.

    Temperature acts on probabilities, because ``attentions`` from transformers
    are already softmaxed; ``temperature=1.0`` is the identity.
    """
    if student.ndim != 3 or teacher.ndim != 3:
        raise ValueError(
            f"student and teacher must be [heads, queries, visual]; got "
            f"{tuple(student.shape)} and {tuple(teacher.shape)}"
        )
    if student.shape[0] != teacher.shape[0] or student.shape[2] != teacher.shape[2]:
        raise ValueError(
            f"head/visual dims must match: student {tuple(student.shape)} vs "
            f"teacher {tuple(teacher.shape)}"
        )
    with torch.no_grad():
        t = _row_averaged_distribution(teacher, temperature).detach()  # [H, V]
    s = _row_averaged_distribution(student, temperature)  # [H, V]
    per_head = F.kl_div(s.log(), t, reduction="none").sum(dim=-1)  # KL(t || s), [H]
    if head_weights is None:
        return per_head.sum()
    if isinstance(head_weights, str):
        weight = _head_weights(teacher, head_weights).to(per_head.dtype)
    else:
        weight = head_weights.to(device=per_head.device, dtype=per_head.dtype)
    return (weight * per_head).sum()


# ----------------------------------------------------------------- training

def visual_scores(
    attention: torch.Tensor, temperature: float = 1.0, head_merge: str = "mean"
) -> torch.Tensor:
    """Inference-time importance ``[V]`` for each visual token, from ``[H, Q, V]``.

    Deliberately the *same* quantity the loss is defined on -- row-average first,
    then renormalise over the visual segment -- so training and inference share
    one ruler. ``head_merge="mean"`` is the agreed merge: the loss treats heads
    symmetrically (a sum of per-head KLs), so a plain mean over heads is the only
    merge that introduces no new free parameters (``"sum"`` differs only by the
    constant factor ``H``; ``"max"`` is kept for ablation).
    """
    per_head = _row_averaged_distribution(attention, temperature)  # [H, V]
    if head_merge == "mean":
        return per_head.mean(dim=0)
    if head_merge == "sum":
        return per_head.sum(dim=0)
    if head_merge == "max":
        return per_head.max(dim=0).values
    raise ValueError(f"unknown head_merge {head_merge!r}")


@dataclass
class Prepared:
    """Device-side cached tensors for one sample."""

    prompt_ids: torch.Tensor        # [1, T]
    prompt_embeds: torch.Tensor     # [1, T, hidden], no grad
    prompt_positions: torch.Tensor  # [3, 1, T], no grad
    teacher_embeds: torch.Tensor    # [1, A, hidden], no grad
    visual_indices: torch.Tensor    # [V]
    image_grid_thw: torch.Tensor | None


class Layer1Trainer:
    """Two trimmed forward passes per step: student (grad) and teacher (no grad)."""

    def __init__(
        self,
        handle: Any,
        samples: list,
        device: torch.device,
        compute_dtype: torch.dtype,
        num_pseudo: int = 32,
        score_layer: int = 1,
        teacher_max_rows: int = 32,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.handle = handle
        self.text = handle.text
        self.device = device
        self.compute_dtype = compute_dtype
        self.score_layer = int(score_layer)
        self.teacher_max_rows = int(teacher_max_rows)
        self.temperature = float(temperature)
        self.image_token_id = int(getattr(handle.config(), "image_token_id", 151655))

        self.scorer = PseudoTokenScorer.from_handle(
            handle,
            num_pseudo=num_pseudo,
            score_layer=self.score_layer,
            temperature=self.temperature,
        ).to(device=device, dtype=compute_dtype)
        for module in (self.text, handle.embed_tokens()):
            module.eval()
        if handle.visual is not None:
            handle.visual.eval()

        self.items: list[Prepared] = []
        for sample in samples:
            prepared = self._prepare(sample)
            if prepared is not None:
                self.items.append(prepared)
        if not self.items:
            raise RuntimeError("no usable samples after preparation")
        self._rng = random.Random(seed)
        self.order = list(range(len(self.items)))
        self._epoch_pos = 0

    # -- preparation -------------------------------------------------------

    def _prepare(self, sample: Any) -> Prepared | None:
        from .read import build_prompt_embeds

        prompt_ids = sample.prompt_ids.to(self.device)
        visual_indices = torch.where(prompt_ids[0] == self.image_token_id)[0].long()
        if visual_indices.numel() == 0:
            return None

        grid = sample.image_grid_thw.to(self.device) if sample.image_grid_thw is not None else None
        pixels = sample.pixel_values.to(self.device) if sample.pixel_values is not None else None
        with torch.no_grad():
            prompt_embeds = build_prompt_embeds(
                self.handle, prompt_ids, pixel_values=pixels, image_grid_thw=grid
            )
            # Real mRoPE positions for the prompt -- the thing the hand-written
            # layer-0 read had to skip entirely.
            prompt_positions = self.handle.wrapper.get_rope_index(
                prompt_ids, image_grid_thw=grid
            )[0]

            answer_ids = sample.answer_ids[0].to(self.device)[: self.teacher_max_rows]
            if answer_ids.numel() == 0:
                return None
            teacher_embeds = self.handle.embed_tokens()(answer_ids).unsqueeze(0)

        return Prepared(
            prompt_ids=prompt_ids,
            prompt_embeds=prompt_embeds,
            prompt_positions=prompt_positions,
            teacher_embeds=teacher_embeds.to(prompt_embeds.dtype),
            visual_indices=visual_indices,
            image_grid_thw=grid,
        )

    # -- forward -----------------------------------------------------------

    def student_distribution(self, prepared: Prepared) -> torch.Tensor:
        """``[H, P, V]`` layer-``score_layer`` scores of the pseudo rows (grad)."""
        pseudo = self.scorer.embed_pseudo.unsqueeze(0).to(self.compute_dtype)
        positions = extend_position_ids(prepared.prompt_positions, int(pseudo.shape[1]))
        return pre_rope_read(
            self.text,
            prepared.prompt_embeds,
            pseudo,
            positions,
            prepared.visual_indices,
            self.score_layer,
        )

    def teacher_distribution(self, prepared: Prepared) -> torch.Tensor:
        """``[H, A, V]`` -- the real answer rows' scores at the same layer."""
        with torch.no_grad():
            positions = extend_position_ids(
                prepared.prompt_positions, int(prepared.teacher_embeds.shape[1])
            )
            return pre_rope_read(
                self.text,
                prepared.prompt_embeds,
                prepared.teacher_embeds,
                positions,
                prepared.visual_indices,
                self.score_layer,
            ).detach()

    def loss_for(self, prepared: Prepared, teacher: torch.Tensor | None = None) -> torch.Tensor:
        student = self.student_distribution(prepared)
        if teacher is None:
            teacher = self.teacher_distribution(prepared)
        return layer1_kl(student, teacher, temperature=self.temperature)

    # -- optim / step ------------------------------------------------------

    def build_optimizer(
        self,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        """AdamW over ``embed_pseudo``.

        ``betas`` is the LookaheadKV setting. ``AdamW`` decays weights in a
        decoupled way, so at the default ``weight_decay=0.0`` this is plain Adam.
        """
        return torch.optim.AdamW(
            [{"params": [self.scorer.embed_pseudo], "lr": lr}],
            weight_decay=weight_decay,
            betas=betas,
        )

    @staticmethod
    def build_scheduler(
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_ratio: float = 0.02,
        min_lr: float = 0.0,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Linear warmup over ``warmup_ratio`` of the run, then cosine to ``min_lr``.

        ``total_steps`` counts optimizer steps (one per batch), and the schedule
        is indexed by *completed* steps -- so the first step already runs at
        ``1/warmup`` of the peak rate rather than at exactly zero, and the last
        step lands on ``min_lr``. Depends only on the optimizer, not on trainer
        state, so it is a static method.
        """
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        peak = float(optimizer.param_groups[0]["lr"])
        if peak <= 0.0:
            raise ValueError(f"peak learning rate must be positive, got {peak}")
        warmup = max(1, int(round(total_steps * warmup_ratio)))

        def decay(step: int) -> float:
            """Fraction of the peak rate at ``step`` completed steps."""
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        if min_lr <= 0.0:
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay)
        # ``LambdaLR`` multiplies the base rate, so a non-zero floor needs the
        # affine map spelled out.
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: ((min_lr + (peak - min_lr) * decay(step)) / peak),
        )

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        batch_ids: list[int],
        max_grad_norm: float = 1.0,
    ) -> dict[str, float]:
        """One optimizer step on the mean loss over ``batch_ids``.

        Samples are accumulated one at a time because the scoring read is batch-1
        (samples have different visual-token counts), and the optimizer steps
        once at the end -- so a large ``batch_size`` costs wall-clock, not memory.
        The divisor is the number of samples actually visited, which makes the
        gradient the true mean over the batch and ``lr`` mean what it says.
        """
        if not batch_ids:
            raise ValueError("train_step needs at least one sample id")
        self.scorer.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for item_id in batch_ids:
            prepared = self.items[item_id]
            # Teachers are prompt-only + frozen, so they can be cached per sample
            # across steps instead of recomputed; kept simple here.
            teacher = self.teacher_distribution(prepared)
            loss = self.loss_for(prepared, teacher=teacher)
            (loss / len(batch_ids)).backward()
            total += float(loss.item())
            del teacher
        parameters = [p for p in self.scorer.parameters() if p.grad is not None]
        # Measured *before* clipping: afterwards it is pinned at ``max_grad_norm``
        # on every step that clipped, which hides how hard the gradient pushed.
        grad_norm = (
            float(torch.cat([p.grad.flatten() for p in parameters]).norm().item())
            if parameters
            else 0.0
        )
        torch.nn.utils.clip_grad_norm_(self.scorer.parameters(), max_grad_norm)
        optimizer.step()
        return {
            "train_kl": total / len(batch_ids),
            "grad_norm": grad_norm,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

    def next_batch_ids(self, batch_size: int) -> list[int]:
        """Shuffled sample ids, reshuffling at each epoch boundary."""
        ids: list[int] = []
        while len(ids) < batch_size:
            if self._epoch_pos >= len(self.order):
                self._rng.shuffle(self.order)
                self._epoch_pos = 0
            ids.append(self.order[self._epoch_pos])
            self._epoch_pos += 1
        return ids

    # -- metrics -----------------------------------------------------------

    def visual_vector(self, attention: torch.Tensor) -> torch.Tensor:
        """``[H, Q, V]`` -> ``[V]`` importance, using the scorer's own settings.

        The same merge inference will use, so the reported ``recall@k`` measures
        the selection the deployed scorer actually makes.
        """
        return visual_scores(
            attention,
            temperature=self.scorer.temperature,
            head_merge=self.scorer.head_merge,
        )

    def metrics_for(self, prepared: Prepared, teacher: torch.Tensor | None = None) -> dict[str, float]:
        """KL + recall@k of the student against the genuine answer-row teacher."""
        from .loss import recall_at_k  # local: keeps ``layer1`` free of train.py
        from .train import _recall_ks

        with torch.no_grad():
            student = self.student_distribution(prepared)
        if teacher is None:
            teacher = self.teacher_distribution(prepared)

        teacher_vec = self.visual_vector(teacher).float()
        student_vec = self.visual_vector(student).float()

        out: dict[str, float] = {
            "kl": float(layer1_kl(student, teacher, temperature=self.temperature).item()),
        }
        for k, value in recall_at_k(teacher_vec, student_vec, ks=_recall_ks(int(teacher_vec.numel()))).items():
            out[f"recall@{k}"] = value

        # Is the KL objective degenerate? KL(t || uniform) is the headroom: near
        # zero means the teacher is flat over the visual tokens and any KL on it
        # reports a number that barely moves. Reported per sample at startup.
        with torch.no_grad():
            flat = _row_averaged_distribution(teacher, self.temperature)  # [H, V]
            entropy = -(flat * flat.clamp_min(1e-8).log()).sum(dim=-1)  # [H]
            out["teacher_kl_uniform"] = float(
                (math.log(flat.shape[-1]) - entropy).mean().item()
            )
        return out

    def average_metrics(
        self, prepared_list: list[Prepared], teacher_cache: dict[int, torch.Tensor] | None = None
    ) -> dict[str, float]:
        """Mean of :meth:`metrics_for` over a held-out list."""
        if not prepared_list:
            return {}
        sums: dict[str, float] = {}
        for index, prepared in enumerate(prepared_list):
            teacher = (teacher_cache or {}).get(index)
            for key, value in self.metrics_for(prepared, teacher=teacher).items():
                sums[key] = sums.get(key, 0.0) + value
        count = len(prepared_list)
        return {key: value / count for key, value in sums.items()}


# -- checkpoints -----------------------------------------------------------

_META_FIELDS = ("num_pseudo", "hidden_size", "score_layer", "temperature", "head_merge")


def save_scorer(scorer: PseudoTokenScorer, path: str, extra: dict | None = None) -> None:
    """Write the scorer plus everything inference needs to read its scores out.

    ``score_layer``/``temperature``/``head_merge`` are part of the checkpoint on
    purpose: they decide the ranking, so inferring them from defaults instead
    would silently prune along a different order.
    """
    payload = {"kind": "pseudo_token", "state_dict": scorer.state_dict(), "extra": extra or {}}
    payload.update({field: getattr(scorer, field) for field in _META_FIELDS})
    torch.save(payload, path)


def load_scorer(path: str, device: Any = None, dtype: torch.dtype | None = None) -> PseudoTokenScorer:
    """Rebuild a :class:`PseudoTokenScorer` written by :func:`save_scorer`."""
    payload = torch.load(str(path), map_location="cpu")
    if payload.get("kind") != "pseudo_token":
        raise ValueError(f"{path} is not a pseudo-token checkpoint (kind={payload.get('kind')!r})")
    missing = [field for field in _META_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"checkpoint {path} is missing {missing}")
    scorer = PseudoTokenScorer(
        int(payload["hidden_size"]),
        num_pseudo=int(payload["num_pseudo"]),
        score_layer=int(payload["score_layer"]),
        temperature=float(payload["temperature"]),
        head_merge=str(payload["head_merge"]),
    )
    scorer.load_state_dict(payload["state_dict"])
    if device is not None or dtype is not None:
        scorer = scorer.to(device=device, dtype=dtype)
    return scorer
