"""Training objectives and retrieval-style agreement metrics.

Loss follows the plan: turn the per-visual-token score vectors (teacher and
student, both produced by :func:`lookhawk.read.score_from_conditioned`) into
categorical distributions over the *same* visual tokens with a softmax at
temperature ``tau``, then take the KL divergence of the student under the
(detached) teacher. ``recall_at_k`` mirrors LookaheadKV's metric: the mean
fraction of the teacher's top-k that the student also keeps.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def _dist_logprobs(
    vector: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    vector = vector.float() / float(temperature)
    return F.log_softmax(vector, dim=-1)


def kl_loss(
    student_vec: torch.Tensor,
    teacher_vec: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL divergence ``D_KL(teacher || student)`` over the visual-token dim.

    Both vectors rank the same set of visual tokens; ``teacher_vec`` is
    detached by the caller. Returns a scalar that is zero when the student
    exactly reproduces the teacher's distribution.
    """
    if student_vec.ndim != 1 or teacher_vec.ndim != 1:
        raise ValueError("student_vec and teacher_vec must be rank-1")
    if student_vec.numel() != teacher_vec.numel():
        raise ValueError("student and teacher vectors must have the same length")
    if student_vec.numel() == 0:
        raise ValueError("cannot take KL over an empty visual-token vector")

    log_student = _dist_logprobs(student_vec, temperature)
    log_teacher = _dist_logprobs(teacher_vec, temperature)
    del eps
    # F.kl_div(input=log p, target=log q, log_target=True) = sum q (log q - log p).
    return F.kl_div(log_student, log_teacher, log_target=True, reduction="sum")


def rank_loss(
    reference: torch.Tensor,
    predicted: torch.Tensor,
    ratios: Iterable[float] = (0.05, 0.10, 0.20),
) -> torch.Tensor:
    """Scale-free bipartite ranking loss over the reference's top-k selection.

    The HAWK-aggregated score vectors are near-uniform (std ~4e-4), so a
    softmax KL on them is ~0 and gradient-free regardless of the query. What
    pruning actually consumes is the *ordering* (top-k keep). This loss
    supervises exactly that: for each budget ``k = round(ratio * V)`` it treats
    the reference's top-k as positives and the rest as negatives and penalises
    every (negative, positive) pair whose predicted scores are inverted,
    ``mean(ReLU(pred_neg - pred_pos))``. Gradients are unit-scale per violated
    pair, independent of the score magnitude, so the near-flat scale does not
    kill the signal.
    """
    if reference.ndim != 1 or predicted.ndim != 1:
        raise ValueError("reference and predicted must be rank-1")
    reference = reference.float()
    predicted = predicted.float()
    length = reference.numel()
    if predicted.numel() != length:
        raise ValueError("reference and predicted must have the same length")
    if length < 2:
        return torch.zeros((), dtype=reference.dtype, device=reference.device)

    terms: list[torch.Tensor] = []
    for ratio in ratios:
        count = max(1, int(round(ratio * length)))
        count = min(count, length - 1)
        pos_idx = reference.topk(count).indices
        pos = predicted[pos_idx]
        neg_mask = torch.ones(length, dtype=torch.bool, device=predicted.device)
        neg_mask[pos_idx] = False
        neg = predicted[neg_mask]
        # ReLU(neg - pos): violated if a negative outranks a positive.
        terms.append((neg[:, None] - pos[None, :]).clamp(min=0.0).mean())
    return torch.stack(terms).mean()


def recall_at_k(
    reference: torch.Tensor,
    predicted: torch.Tensor,
    ks: Iterable[int],
) -> dict[int, float]:
    """Mean top-k overlap between two score vectors' orderings.

    ``reference`` and ``predicted`` are rank-1 vectors of identical length V.
    Returns ``{k: recall}`` where recall = |topk_ref ∩ topk_pred| / k.
    """
    if reference.ndim != 1 or predicted.ndim != 1:
        raise ValueError("reference and predicted must be rank-1")
    reference = reference.float()
    predicted = predicted.float()
    length = reference.numel()
    if predicted.numel() != length:
        raise ValueError("reference and predicted must have the same length")

    _, ref_top = reference.topk(length, largest=True)
    _, pred_top = predicted.topk(length, largest=True)

    # Position of each visual token in each ordering.
    ref_rank = torch.empty(length, dtype=torch.long, device=reference.device)
    pred_rank = torch.empty(length, dtype=torch.long, device=reference.device)
    ref_rank.scatter_(0, ref_top, torch.arange(length, device=reference.device))
    pred_rank.scatter_(0, pred_top, torch.arange(length, device=reference.device))

    out: dict[int, float] = {}
    for k in ks:
        if k < 1:
            continue
        k = min(int(k), length)
        hit = (ref_rank < k) & (pred_rank < k)
        out[k] = float(hit.sum().item()) / float(k)
    return out
