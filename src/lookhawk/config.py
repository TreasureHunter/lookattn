"""Configuration for the layer-0 lookahead scorer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

TEACHER_SELECTION_VALUES = frozenset({"first", "strided"})
LoraTarget = str  # "q" | "k" | "v" | "o" (attention projections)


@dataclass(frozen=True)
class LookhawkConfig:
    """Hyper-parameters for training the lookahead scorer.

    ``lookahead_size`` is the number of learned appended rows the student and
    the inference-time scorer use. ``teacher_max_rows`` bounds how many real
    answer tokens form the teacher's conditioning block (teacher rows see
    earlier answer tokens causally, which a prompt-only student can never see,
    so unbounded teacher rows push the target away from what is learnable).

    LoRA scaling follows LookaheadKV: ``y += (alpha / rank) * x @ A @ B`` with
    ``A ~ N(0, 0.01)`` and ``B = 0``, so the scorer is a no-op at init.
    """

    # Architecture
    lookahead_size: int = 32
    teacher_max_rows: int = 32
    teacher_selection: str = "first"  # "first" or "strided"
    temperature: float = 1.0

    # LoRA on the projections of decoder layer 0 *inside* the conditioning
    # read, applied to appended rows only (subset of {"q","k","v","o"}). These
    # are secondary: the steerable part that actually decides the selection is
    # ``score_lora_targets`` below. Empty by default (minimal params).
    lora_rank: int = 8
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ()

    # Score-stage steering: additive LoRA applied to the read-conditioned
    # appended hidden states *at scoring time* (the ``q`` used for the visual
    # dot-product). This is what makes the learned parameters genuinely move the
    # output; the read-conditioned input keeps it sample-dependent. Default "q"
    # is the minimal steering; "k"/"v"/"o" are unused at the score stage.
    score_lora_targets: tuple[str, ...] = ("q",)

    # Optional layer-0 MLP LoRA (gate/up/down); empty by default.
    mlp_lora_targets: tuple[str, ...] = ()

    embed_scale: float = 0.02

    # Scoring / aggregation (mirrors HawkConfig).
    score_normalization: str = "l2"
    head_weights_path: str | None = None

    # Training objective. "rank" supervises the teacher's top-k selection with a
    # scale-free bipartite ranking loss over the flat HAWK-aggregated scores
    # (which a softmax KL cannot see). "kl" keeps the old temperature KL.
    loss: str = "rank"
    rank_ratios: tuple[float, ...] = (0.05, 0.10, 0.20)  # budgets of V for rank loss

    # Data
    max_seq_len: int = 2048

    def __post_init__(self) -> None:
        if self.lookahead_size < 1:
            raise ValueError("lookahead_size must be at least 1")
        if self.teacher_max_rows < 1:
            raise ValueError("teacher_max_rows must be at least 1")
        if self.teacher_selection not in TEACHER_SELECTION_VALUES:
            raise ValueError(
                f"teacher_selection must be one of {sorted(TEACHER_SELECTION_VALUES)}, "
                f"got {self.teacher_selection}"
            )
        valid = frozenset({"q", "k", "v", "o"})
        if not set(self.lora_targets) <= valid:
            raise ValueError(f"lora_targets must be a subset of {sorted(valid)}")
        if not set(self.score_lora_targets) <= valid:
            raise ValueError(
                f"score_lora_targets must be a subset of {sorted(valid)}"
            )
        valid_mlp = frozenset({"gate", "up", "down"})
        if not set(self.mlp_lora_targets) <= valid_mlp:
            raise ValueError(
                f"mlp_lora_targets must be a subset of {sorted(valid_mlp)}"
            )
        if self.score_normalization not in {"l2", "softmax", "minmax"}:
            raise ValueError(
                f"score_normalization must be l2, softmax, or minmax; "
                f"got {self.score_normalization}"
            )
        if not 0.0 < self.embed_scale:
            raise ValueError("embed_scale must be positive")
        if self.loss not in {"rank", "kl"}:
            raise ValueError(f"loss must be 'rank' or 'kl', got {self.loss!r}")
        if any(not 0.0 < ratio < 1.0 for ratio in self.rank_ratios):
            raise ValueError("rank_ratios entries must lie in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
