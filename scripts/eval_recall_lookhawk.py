#!/usr/bin/env python3
"""Offline recall@k / KL report: teacher (answer) vs student vs HAWK baseline.

For every row of a held-out jsonl this computes three visual-score vectors on
the *same* prompt:

* teacher  -- the genuine target: real answer-token rows queried pre-attention
  through HAWK's exact aggregation (no conditioning read);
* student  -- the trained lookahead rows (when ``--scorer`` is given);
* hawk     -- HAWK's original instruction-text query rows (training-free
  baseline that already shares the same scoring function).

The report averages KL(student||teacher), recall@k(student-topk vs teacher-topk),
and the same pair for hawk-vs-teacher, so you can see the trained scorer beat
the training-free baseline before any benchmark run.

Run under the project runtime::

    PYTHON_BIN="$(which python)" bash scripts/run.sh \
        python scripts/eval_recall_lookhawk.py \
            --scorer outputs/lookhawk_smoke/scorer.pt \
            --jsonl data/smoke/val.jsonl --no-vision
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from lookhawk.config import LookhawkConfig
from lookhawk.train import (
    ScorerTrainer,
    default_head_weights,
    encode_jsonl,
    load_and_trim_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--jsonl", default="data/smoke/val.jsonl")
    parser.add_argument("--scorer", default=None, help="optional scorer checkpoint")
    parser.add_argument("--head-weights", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--no-vision", action="store_true", help="skip the vision tower (fast smoke)")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = build_parser()
    from transformers import AutoProcessor

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        dtype = dtype_map[args.dtype]

    checkpoint = None
    if args.scorer:
        checkpoint = torch.load(args.scorer, map_location="cpu")
        cfg = LookhawkConfig(**checkpoint["config"])
    else:
        cfg = LookhawkConfig()

    head_weights = None
    if args.head_weights:
        from lookhawk.train import load_head_weights

        head_weights = load_head_weights(args.head_weights)
    elif checkpoint is not None and checkpoint.get("head_weights") is not None:
        head_weights = torch.tensor(checkpoint["head_weights"], dtype=torch.float32)
    else:
        head_weights = default_head_weights()

    processor_kwargs = {}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, **processor_kwargs)
    samples, skips = encode_jsonl(
        processor, args.jsonl, max_seq_len=args.max_seq_len, limit=args.limit
    )
    print(f"[data] {len(samples)} samples from {args.jsonl} (skips={skips})")
    if not samples:
        raise SystemExit("no samples")

    t0 = time.time()
    handle = load_and_trim_model(args.model_path, device, dtype)
    print(f"[model] loaded on {device} ({time.time() - t0:.1f}s)")

    trainer = ScorerTrainer(
        handle,
        cfg,
        samples,
        head_weights=head_weights,
        device=device,
        compute_dtype=dtype,
        use_vision=not args.no_vision,
        seed=0,
    )
    print(f"[prepare] {len(trainer.items)}/{len(samples)} prepared on device")

    if checkpoint is not None:
        trainer.scorer.load_state_dict(checkpoint["state_dict"])
        label = f"scorer {args.scorer}"
    else:
        label = "untrained scorer (init)"
    print(f"[scorer] using {label}")

    metrics = trainer.average_metrics(trainer.items)
    avg_visual = sum(int(p.teacher_vec.numel()) for p in trainer.items) / max(1, len(trainer.items))
    print(f"[summary] samples={len(trainer.items)} avg_visual_tokens={avg_visual:.1f}")

    order = [1, 2, 4, 8, 16, 32, 64, 128]
    print("\nteacher-vs-student (genuine pre-read teacher):")
    for k in order:
        value = metrics.get(f"recall@{k}")
        if value is not None:
            print(f"  recall@{k:<4d} = {value:.3f}")
    print(f"  kl                = {metrics.get('kl', float('nan')):.4f}")
    print(f"  rank_loss         = {metrics.get('rank_loss', float('nan')):.4f}")
    print("\nteacher-vs-hawk (training-free instruction-query baseline):")
    for k in order:
        value = metrics.get(f"hawk_recall@{k}")
        if value is not None:
            print(f"  recall@{k:<4d} = {value:.3f}")
    print(f"  kl                = {metrics.get('hawk_kl_vs_teacher', float('nan')):.4f}")
    print(f"  rank_loss         = {metrics.get('hawk_rank_loss', float('nan')):.4f}")
    print(f"\nfull metric set: { {k: round(v, 4) for k, v in sorted(metrics.items())} }")


if __name__ == "__main__":
    main()
