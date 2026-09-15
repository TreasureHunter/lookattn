"""Trainable modules: additive LoRA and the layer-0 lookahead scorer.

The scorer owns the only learned parameters in the whole approach:

* ``embed_lookahead`` -- ``lookahead_size`` row embeddings appended after the
  prompt at training and inference time (the "future-token" stand-ins).
* optional additive LoRAs on decoder layer-0 projections, applied only to the
  appended rows during the conditioning read. Default targets are ``q``/``k``
  (the minimal adaptation), matching LookaheadKV's ``LlamaLoRA`` form.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import LookhawkConfig


class LoraLinearAdd(nn.Module):
    """Low-rank additive projection ``y = (alpha / rank) * x @ A @ B``.

    ``A`` is drawn ~ N(0, 0.01) and ``B`` is zero-initialized, so the module
    contributes nothing at init (LookaheadKV ``LlamaLoRA`` semantics).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.0,
        init_a_std: float = 0.01,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be at least 1")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.randn(self.input_dim, self.rank) * init_a_std)
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.output_dim))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaling * ((self.dropout(x) @ self.lora_A) @ self.lora_B)

    def extra_repr(self) -> str:
        return (
            f"{self.input_dim}->{self.output_dim}, rank={self.rank}, "
            f"scaling={self.scaling:.3g}"
        )


_ATTN_LORA = frozenset({"q", "k", "v", "o"})
_MLP_LORA = frozenset({"gate", "up", "down"})


class Layer0Scorer(nn.Module):
    """Learnable lookahead rows + additive LoRA over decoder layer 0."""

    def __init__(
        self,
        cfg: LookhawkConfig,
        hidden_size: int,
        attention_out_dims: dict[str, int],
        mlp_dims: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = int(hidden_size)
        rows = cfg.lookahead_size
        self.embed_lookahead = nn.Parameter(
            torch.randn(rows, self.hidden_size) * cfg.embed_scale
        )
        for name in cfg.lora_targets:
            if name not in _ATTN_LORA:
                raise ValueError(f"unknown attention LoRA target {name!r}")
            out_dim = attention_out_dims[name]
            setattr(
                self,
                f"{name}_lora",
                LoraLinearAdd(
                    self.hidden_size,
                    out_dim,
                    rank=cfg.lora_rank,
                    alpha=cfg.lora_alpha,
                    dropout=cfg.lora_dropout,
                ),
            )
        for name in cfg.score_lora_targets:
            if name not in _ATTN_LORA:
                raise ValueError(f"unknown score LoRA target {name!r}")
            setattr(
                self,
                f"score_{name}_lora",
                LoraLinearAdd(
                    self.hidden_size,
                    attention_out_dims[name],
                    rank=cfg.lora_rank,
                    alpha=cfg.lora_alpha,
                    dropout=cfg.lora_dropout,
                ),
            )
        for name in cfg.mlp_lora_targets:
            if name not in _MLP_LORA:
                raise ValueError(f"unknown MLP LoRA target {name!r}")
            in_dim, out_dim = mlp_dims[name]
            setattr(
                self,
                f"{name}_lora",
                LoraLinearAdd(
                    in_dim,
                    out_dim,
                    rank=cfg.lora_rank,
                    alpha=cfg.lora_alpha,
                    dropout=cfg.lora_dropout,
                ),
            )

    @classmethod
    def from_layer0(cls, layer0: Any, cfg: LookhawkConfig) -> "Layer0Scorer":
        """Build a scorer with projection shapes read from decoder layer 0."""
        attention = layer0.self_attn
        hidden = int(attention.q_proj.in_features)

        def out_features(projection: Any) -> int:
            if hasattr(projection, "out_features"):
                return int(projection.out_features)
            return int(projection.in_features)

        attention_out_dims = {
            "q": out_features(attention.q_proj),
            "k": out_features(attention.k_proj),
            "v": out_features(attention.v_proj),
            "o": out_features(attention.o_proj),
        }
        mlp_dims: dict[str, tuple[int, int]] = {}
        if cfg.mlp_lora_targets:
            mlp = layer0.mlp
            intermediate = out_features(mlp.gate_proj)
            mlp_dims = {
                "gate": (hidden, intermediate),
                "up": (hidden, intermediate),
                "down": (intermediate, hidden),
            }
        return cls(cfg, hidden, attention_out_dims, mlp_dims)

    def proj_lora(self, name: str) -> LoraLinearAdd | None:
        """Return the named read-stage LoRA module or ``None`` if disabled."""
        module = getattr(self, f"{name}_lora", None)
        if module is None:
            return None
        if not isinstance(module, LoraLinearAdd):
            raise TypeError(f"{name}_lora is not a LoraLinearAdd")
        return module

    def score_lora(self, name: str) -> LoraLinearAdd | None:
        """Return the named score-stage steering LoRA or ``None`` if disabled."""
        module = getattr(self, f"score_{name}_lora", None)
        if module is None:
            return None
        if not isinstance(module, LoraLinearAdd):
            raise TypeError(f"score_{name}_lora is not a LoraLinearAdd")
        return module

    def extra_repr(self) -> str:
        loras = sorted(self.cfg.lora_targets) + sorted(self.cfg.mlp_lora_targets)
        score = sorted(self.cfg.score_lora_targets)
        return (
            f"lookahead_size={self.cfg.lookahead_size}, "
            f"hidden={self.hidden_size}, read_lora=[{','.join(loras) or '-'}], "
            f"score_lora=[{','.join(score) or '-'}]"
        )
