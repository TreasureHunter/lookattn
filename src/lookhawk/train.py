"""Training loop for the layer-0 lookahead scorer.

Per sample the teacher and the student are both visual-token score vectors from
HAWK's exact aggregation, but from different query sources:

* teacher -- the real answer-token rows, queried *pre-attention* (no read).
  Their ``q`` is aggregated against the prompt's pre-RoPE visual keys exactly
  like HAWK. A conditioning read would wash the answer content out, so this
  genuine target is read-free (see the read-degeneracy analysis).
* student -- the learned appended rows are read through decoder layer 0
  (:func:`~lookhawk.read.conditioned_layer0_read`) and then scored; a
  score-stage q LoRA (:func:`~lookhawk.read.score_from_conditioned` with
  ``scorer``) steers the read-conditioned query toward the answer's. This
  steering is what makes the parameters actually move the selection.

The default loss is a scale-free bipartite ranking loss over the teacher's top-k
selection (:func:`lookhawk.loss.rank_loss`), because the HAWK-aggregated score
vectors are near-uniform (std ~4e-4) and a softmax KL on them is ~0. ``cfg.loss
= "kl"`` restores the temperature-KL objective.

Only the scorer's parameters are trainable; the Qwen2.5-VL weights are frozen
and every non-essential module is dropped after load. The prompt embeddings and
teacher score vectors are computed once per sample and cached, so each step is
just one student read + backward over decoder layer 0.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from hawk.config import QWEN2_5_VL_7B_HEAD_WEIGHTS
from hawk.pruning import _query_indices

from .config import LookhawkConfig
from .data import Sample, SkipRow, encode_rows, iter_rows
from .loss import kl_loss, rank_loss, recall_at_k
from .model_graph import (
    IM_END_TOKEN_ID,
    IMAGE_TOKEN_ID,
    VISION_END_TOKEN_ID,
    ModelHandle,
    resolve_handle,
    verify_model_graph,
)
from .modules import Layer0Scorer
from .read import (
    build_prompt_embeds,
    conditioned_layer0_read,
    score_from_conditioned,
    score_from_query_embeds,
    select_teacher_answer_ids,
)


def default_head_weights() -> torch.Tensor:
    return torch.tensor(QWEN2_5_VL_7B_HEAD_WEIGHTS, dtype=torch.float32)


def load_head_weights(path: str | None) -> torch.Tensor:
    """Read the 28-entry head-weight vector from a json/np/torch file or default."""
    if path is None:
        return default_head_weights()
    blob = Path(path).read_text(encoding="utf-8")
    import numpy as np  # optional dep only when a numpy file is given

    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu").float().reshape(-1)
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path)).float().reshape(-1)
    data = json.loads(blob) if path.endswith(".json") else [float(x) for x in blob.split()]
    return torch.tensor(data, dtype=torch.float32).reshape(-1)


def load_and_trim_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    keep_layers: int = 1,
) -> ModelHandle:
    """Load the Qwen2.5-VL model, freeze it, and keep only the first ``keep_layers``.

    Returns a :class:`ModelHandle` over the trimmed graph. Only the vision
    tower, token embedding, and decoder layers ``0 .. keep_layers-1`` remain
    resident; the rest and the LM head are released. This keeps the resident
    footprint to a few GB while training.

    ``keep_layers=2`` is what the layer-1 attention-KL path needs: the pseudo
    rows are read by layer 0 and scored by layer 1's attention.
    """
    from transformers import AutoModelForVision2Seq

    outer = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    for parameter in outer.parameters():
        parameter.requires_grad_(False)

    handle = resolve_handle(outer)
    verify_model_graph(handle)

    decoder = handle.text
    if len(decoder.layers) > keep_layers:
        kept = list(decoder.layers[:keep_layers])
        decoder.layers = nn.ModuleList(kept)  # drop the rest
        decoder.config.num_hidden_layers = keep_layers
    del outer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return handle


def encode_jsonl(
    processor: Any,
    jsonl_path: str,
    max_seq_len: int = 2048,
    limit: int | None = None,
    workers: int = 1,
) -> tuple[list[Sample], dict[str, int]]:
    """Encode a jsonl split into CPU samples, counting skip reasons.

    ``workers`` parallelizes the encode across processes. It must be combined
    with a CUDA-free moment: the pool is forked, so call this before the model
    is moved onto the GPU.
    """
    skip_log: dict[str, int] = {}
    samples = list(
        encode_rows(
            processor,
            iter_rows(jsonl_path),
            max_seq_len=max_seq_len,
            limit=limit,
            skip_log=skip_log,
            workers=workers,
        )
    )
    return samples, skip_log


@dataclass
class Prepared:
    """Device-side cached tensors for one training sample."""

    prompt_ids: torch.Tensor  # [1, T] on device (token ids for query locator)
    prompt_embeds: torch.Tensor  # [1, T, hidden], no grad
    visual_indices: torch.Tensor  # [V] positions of image tokens in the prompt
    teacher_vec: torch.Tensor  # [V] fp32, teacher's aggregated visual scores


def _image_token_id(handle: ModelHandle) -> int:
    return int(getattr(handle.config(), "image_token_id", IMAGE_TOKEN_ID))


class ScorerTrainer:
    """Owns the scorer, optimizer, and per-sample device caches."""

    def __init__(
        self,
        handle: ModelHandle,
        cfg: LookhawkConfig,
        samples: list[Sample],
        head_weights: torch.Tensor,
        device: torch.device,
        compute_dtype: torch.dtype,
        use_vision: bool = True,
        seed: int = 0,
    ) -> None:
        self.handle = handle
        self.cfg = cfg
        self.device = device
        self.compute_dtype = compute_dtype
        self.use_vision = use_vision
        self.image_token_id = _image_token_id(handle)
        self.head_weights = head_weights.to(device=device, dtype=torch.float32)
        self.layer0 = handle.layer0()

        self.scorer = Layer0Scorer.from_layer0(self.layer0, cfg).to(
            device=device, dtype=compute_dtype
        )
        for module in (self.layer0, handle.embed_tokens()):
            module.eval()
        if handle.visual is not None:
            handle.visual.eval()

        self._rng = random.Random(seed)
        self.items: list[Prepared] = []
        for sample in samples:
            prepared = self._prepare(sample)
            if prepared is not None:
                self.items.append(prepared)
        if not self.items:
            raise RuntimeError("no usable training samples after encoding/preparation")
        self.order = list(range(len(self.items)))
        self._epoch_pos = 0

    # -- preparation -------------------------------------------------------

    def _prepare(self, sample: Sample) -> Prepared | None:
        prompt_ids = sample.prompt_ids.to(self.device)
        visual_indices = torch.where(prompt_ids[0] == self.image_token_id)[0].long()
        if visual_indices.numel() == 0:
            return None

        prompt_embeds = build_prompt_embeds(
            self.handle,
            prompt_ids,
            pixel_values=(
                sample.pixel_values.to(self.device)
                if (self.use_vision and sample.pixel_values is not None)
                else None
            ),
            image_grid_thw=(
                sample.image_grid_thw.to(self.device)
                if (self.use_vision and sample.image_grid_thw is not None)
                else None
            ),
        )

        with torch.no_grad():
            answer_ids = sample.answer_ids[0].to(self.device)
            teacher_ids = select_teacher_answer_ids(answer_ids, self.cfg)
            if teacher_ids.numel() == 0:
                return None
            # Genuine target: the real answer rows' pre-attention queries
            # aggregated exactly like HAWK. No conditioning read -- a read would
            # wash the answer content out (rows collapse to a prompt summary).
            answer_embeds = self.handle.embed_tokens()(teacher_ids).unsqueeze(0)
            teacher_vec = score_from_query_embeds(
                self.layer0,
                prompt_embeds,
                visual_indices,
                answer_embeds,
                self.head_weights,
                normalization=self.cfg.score_normalization,
            ).float().detach()

        return Prepared(
            prompt_ids=prompt_ids,
            prompt_embeds=prompt_embeds,
            visual_indices=visual_indices,
            teacher_vec=teacher_vec,
        )

    # -- forward -----------------------------------------------------------

    def student_vector(self, prepared: Prepared) -> torch.Tensor:
        """Conditioned lookahead rows -> aggregated visual scores [V] (grad)."""
        rows = self.cfg.lookahead_size
        seq = torch.cat(
            [prepared.prompt_embeds, self.scorer.embed_lookahead.unsqueeze(0)], dim=1
        )
        hidden = conditioned_layer0_read(
            self.layer0, seq, appended_size=rows, scorer=self.scorer
        )
        return score_from_conditioned(
            self.layer0,
            prepared.prompt_embeds,
            prepared.visual_indices,
            hidden,
            self.head_weights,
            normalization=self.cfg.score_normalization,
            scorer=self.scorer,
        ).float()

    def kl_for(self, student_vec: torch.Tensor, prepared: Prepared) -> torch.Tensor:
        """Mean KL over the visual tokens (sum-reduction normalised by V)."""
        count = max(1, int(prepared.teacher_vec.numel()))
        return kl_loss(
            student_vec, prepared.teacher_vec, temperature=self.cfg.temperature
        ) / count

    def loss_for(self, student_vec: torch.Tensor, prepared: Prepared) -> torch.Tensor:
        """Training objective routed by ``cfg.loss``.

        ``rank`` is the scale-free bipartite loss over the teacher's top-k (the
        default; the near-uniform HAWK scores make softmax KL ~0). ``kl`` keeps
        the temperature KL for comparison.
        """
        if self.cfg.loss == "rank":
            return rank_loss(
                prepared.teacher_vec, student_vec, ratios=self.cfg.rank_ratios
            )
        return self.kl_for(student_vec, prepared)

    # -- metrics -----------------------------------------------------------

    def metrics_for(self, prepared: Prepared) -> dict[str, float]:
        """KL + recall@k of the current student against teacher and HAWK base."""
        with torch.no_grad():
            student_vec = self.student_vector(prepared).float()
        teacher = prepared.teacher_vec.float()
        ks = _recall_ks(int(teacher.numel()))
        recall = recall_at_k(teacher, student_vec, ks=ks)
        out: dict[str, float] = {
            "kl": float(self.kl_for(student_vec, prepared).item()),
            "rank_loss": float(
                rank_loss(teacher, student_vec, ratios=self.cfg.rank_ratios).item()
            ),
        }
        out.update({f"recall@{k}": v for k, v in recall.items()})

        hawk_vec = self.hawk_instruction_vector(prepared)
        out["hawk_kl_vs_teacher"] = float(
            kl_loss(teacher, hawk_vec, temperature=self.cfg.temperature).item()
            / max(1, int(teacher.numel()))
        )
        out["hawk_rank_loss"] = float(
            rank_loss(teacher, hawk_vec, ratios=self.cfg.rank_ratios).item()
        )
        hawk_recall = recall_at_k(teacher, hawk_vec, ks=ks)
        out.update({f"hawk_recall@{k}": v for k, v in hawk_recall.items()})
        return out

    def hawk_instruction_vector(self, prepared: Prepared) -> torch.Tensor:
        """HAWK's own instruction-query score vector (training-free baseline).

        Shares the exact ``score_from_query_embeds`` transform used for the
        genuine teacher, but with the instruction-text token rows as queries --
        so student, teacher, and the HAWK baseline are all the same function of
        different query rows.
        """
        queries = _query_indices(
            prepared.prompt_ids,
            prepared.visual_indices,
            VISION_END_TOKEN_ID,
            IM_END_TOKEN_ID,
        )
        query_ids = prepared.prompt_ids[0][queries]
        query_embeds = self.handle.embed_tokens()(query_ids).unsqueeze(0)
        return score_from_query_embeds(
            self.layer0,
            prepared.prompt_embeds,
            prepared.visual_indices,
            query_embeds,
            self.head_weights,
            normalization=self.cfg.score_normalization,
        ).float()

    # -- optim / step ------------------------------------------------------

    def build_optimizer(
        self,
        lr_embed: float = 3e-4,
        lr_lora: float = 1e-4,
        weight_decay: float = 0.05,
    ) -> torch.optim.Optimizer:
        embed_group = {"params": [self.scorer.embed_lookahead], "lr": lr_embed}
        lora_params: list[nn.Parameter] = []
        for name, parameter in self.scorer.named_parameters():
            if name != "embed_lookahead":
                lora_params.append(parameter)
        lora_group = {"params": lora_params, "lr": lr_lora}
        return torch.optim.AdamW(
            [embed_group, lora_group],
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

    def next_batch_ids(self, batch_size: int) -> list[int]:
        ids: list[int] = []
        while len(ids) < batch_size:
            if self._epoch_pos >= len(self.order):
                self._rng.shuffle(self.order)
                self._epoch_pos = 0
            ids.append(self.order[self._epoch_pos])
            self._epoch_pos += 1
        return ids

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        batch_ids: list[int],
        grad_accum: int = 1,
        max_grad_norm: float = 1.0,
    ) -> dict[str, float]:
        self.scorer.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_kl = 0.0
        for position, item_id in enumerate(batch_ids):
            prepared = self.items[item_id]
            student_vec = self.student_vector(prepared)
            loss = self.loss_for(student_vec, prepared)
            (loss / grad_accum).backward()
            total_loss += float(loss.item())
            total_kl += float(self.kl_for(student_vec, prepared).item())
        torch.nn.utils.clip_grad_norm_(self.scorer.parameters(), max_grad_norm)
        optimizer.step()

        grad_norm = 0.0
        grads = [p.grad for p in self.scorer.parameters() if p.grad is not None]
        if grads:
            grad_norm = float(torch.cat([g.flatten() for g in grads]).norm().item())
        count = max(1, len(batch_ids))
        return {"train_loss": total_loss / count, "train_kl": total_kl / count,
                "grad_norm": grad_norm}

    def average_metrics(self, prepared_list: list[Prepared]) -> dict[str, float]:
        if not prepared_list:
            return {}
        sums: dict[str, float] = {}
        for prepared in prepared_list:
            for key, value in self.metrics_for(prepared).items():
                sums[key] = sums.get(key, 0.0) + value
        count = len(prepared_list)
        return {key: value / count for key, value in sums.items()}


def _recall_ks(count: int) -> list[int]:
    """A small fixed set of top-k targets adapted to the visual count."""
    if count <= 1:
        return [1]
    raw = [1, 2, 4, 8, 16, 32, 64, 128]
    for ratio in (0.05, 0.10, 0.20):
        raw.append(max(1, math.ceil(count * ratio)))
    ks: list[int] = []
    for k in raw:
        if 1 <= k <= count and k not in ks:
            ks.append(k)
    return sorted(ks)


# -- checkpoint helpers ----------------------------------------------------

def save_scorer(
    scorer: Layer0Scorer,
    cfg: LookhawkConfig,
    path: str | Path,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "config": cfg.to_dict(),
        "state_dict": scorer.state_dict(),
        "head_weights": default_head_weights().tolist(),
        "extra": extra or {},
    }
    torch.save(payload, str(path))


def load_scorer_from_layer0(layer0: Any, path: str | Path) -> Layer0Scorer:
    """Rebuild a scorer from a saved checkpoint and restore its weights."""
    payload = torch.load(str(path), map_location="cpu")
    cfg = LookhawkConfig(**payload["config"])
    scorer = Layer0Scorer.from_layer0(layer0, cfg)
    scorer.load_state_dict(payload["state_dict"])
    return scorer


def load_head_weights_from_checkpoint(path: str | Path) -> torch.Tensor:
    payload = torch.load(str(path), map_location="cpu")
    return torch.tensor(payload.get("head_weights"), dtype=torch.float32)
