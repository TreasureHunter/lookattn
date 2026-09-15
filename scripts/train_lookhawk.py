#!/usr/bin/env python3
"""Train the layer-0 lookahead scorer (Qwen2.5-VL, LoRA on decoder layer 0).

Run from the project root under the HAWK runtime::

    PYTHON_BIN="$(which python)" bash scripts/run.sh \
        scripts/train_lookhawk.py --out-dir outputs/lookhawk_smoke

The whole Qwen2.5-VL model is frozen and trimmed to (vision tower + token
embedding + decoder layer 0); only the scorer (``embed_lookahead`` + q/k LoRA)
is trained against the teacher answer-token scoring described in
``src/lookhawk/train.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from lookhawk.config import LookhawkConfig, TEACHER_SELECTION_VALUES
from lookhawk.data import SkipRow
from lookhawk.train import (
    ScorerTrainer,
    encode_jsonl,
    load_and_trim_model,
    load_head_weights,
    save_scorer,
)


def _parse_lora_targets(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    return values


def _parse_ratios(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one ratio")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train-jsonl", default="data/smoke/train.jsonl")
    parser.add_argument("--val-jsonl", default="data/smoke/val.jsonl")
    parser.add_argument("--out-dir", default="outputs/lookhawk_smoke")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"]
    )
    parser.add_argument("--no-vision", action="store_true", help="skip the vision tower (fast smoke)")
    parser.add_argument("--seed", type=int, default=0)

    # Architecture / data
    parser.add_argument("--lookahead-size", type=int, default=None)
    parser.add_argument("--teacher-max-rows", type=int, default=None)
    parser.add_argument("--teacher-selection", default=None, choices=sorted(TEACHER_SELECTION_VALUES))
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--lora-targets", default=None, type=_parse_lora_targets)
    parser.add_argument("--score-lora-targets", default=None, type=_parse_lora_targets)
    parser.add_argument("--loss", default=None, choices=["rank", "kl"])
    parser.add_argument("--rank-ratios", default=None, type=_parse_ratios)
    parser.add_argument("--score-normalization", default=None, choices=["l2", "softmax", "minmax"])
    parser.add_argument("--head-weights", default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)

    # Training
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr-embed", type=float, default=3e-4)
    parser.add_argument("--lr-lora", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-best", action="store_true", help="save best-val-KL checkpoint")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from transformers import AutoProcessor

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if args.dtype == "auto":
        compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        compute_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[args.dtype]

    processor_kwargs = {}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, **processor_kwargs)

    max_seq_len = args.max_seq_len or 2048
    print(f"[phase] encoding train rows (limit={args.limit_train})...", flush=True)
    train_samples, train_skips = encode_jsonl(
        processor, args.train_jsonl, max_seq_len=max_seq_len, limit=args.limit_train
    )
    val_samples, val_skips = encode_jsonl(
        processor, args.val_jsonl, max_seq_len=max_seq_len, limit=args.limit_val
    )
    print(
        f"[data] train={len(train_samples)} val={len(val_samples)} "
        f"(skips train={train_skips}, val={val_skips})"
    )
    if not train_samples:
        raise SystemExit("no training samples encoded; aborting")

    kwargs: dict = {}
    overrides = {
        "lookahead_size": args.lookahead_size,
        "teacher_max_rows": args.teacher_max_rows,
        "teacher_selection": args.teacher_selection,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "temperature": args.temperature,
        "lora_targets": args.lora_targets,
        "score_lora_targets": args.score_lora_targets,
        "loss": args.loss,
        "rank_ratios": args.rank_ratios,
        "score_normalization": args.score_normalization,
        "max_seq_len": max_seq_len,
    }
    for name, value in overrides.items():
        if value is not None:
            kwargs[name] = value
    cfg = LookhawkConfig(**kwargs)
    print("[cfg]", json.dumps(cfg.to_dict()))

    head_weights = load_head_weights(args.head_weights)
    if head_weights.numel() != 28:
        raise SystemExit(f"expected 28 head weights, got {head_weights.numel()}")

    t0 = time.time()
    print("[phase] loading & trimming model (this is the slow part)...", flush=True)
    handle = load_and_trim_model(args.model_path, device, compute_dtype)
    print(f"[model] trimmed & loaded on {device} ({time.time() - t0:.1f}s)", flush=True)

    print("[phase] preparing per-sample teacher caches...", flush=True)
    trainer = ScorerTrainer(
        handle,
        cfg,
        train_samples,
        head_weights=head_weights,
        device=device,
        compute_dtype=compute_dtype,
        use_vision=not args.no_vision,
        seed=args.seed,
    )
    print(
        f"[train] prepared {len(trainer.items)}/{len(train_samples)} samples on device",
        flush=True,
    )

    optimizer = trainer.build_optimizer(
        lr_embed=args.lr_embed, lr_lora=args.lr_lora, weight_decay=args.weight_decay
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    # Prepare evaluation subsets once (device side).
    val_items: list = []
    for sample in val_samples:
        prepared = trainer._prepare(sample)
        if prepared is not None:
            val_items.append(prepared)
    print(f"[val] prepared {len(val_items)}/{len(val_samples)} val samples", flush=True)

    best_val = float("inf")
    best_key = "rank_loss" if cfg.loss == "rank" else "kl"
    global_step = 0
    for step in range(1, args.steps + 1):
        batch_ids = trainer.next_batch_ids(args.batch_size)
        metrics = trainer.train_step(
            optimizer, batch_ids, grad_accum=args.grad_accum, max_grad_norm=args.max_grad_norm
        )
        global_step += 1
        if step == 1 or step % args.log_every == 0:
            print(
                f"[step {step}/{args.steps}] train_loss={metrics['train_loss']:.6f} "
                f"grad_norm={metrics['grad_norm']:.6f} train_kl={metrics['train_kl']:.6f}"
            )
        if args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            val_metrics = trainer.average_metrics(val_items)
            if val_metrics:
                bits = " ".join(
                    f"{k}={v:.3f}"
                    for k, v in val_metrics.items()
                    if k
                    in (
                        "rank_loss",
                        "kl",
                        "recall@1",
                        "recall@2",
                        "recall@16",
                        "hawk_rank_loss",
                        "hawk_kl_vs_teacher",
                        "hawk_recall@16",
                    )
                )
                print(f"[eval step {step}] {bits}")
                if val_metrics[best_key] < best_val and args.save_best:
                    best_val = val_metrics[best_key]
                    save_scorer(
                        trainer.scorer, cfg, out_dir / "scorer_best.pt",
                        extra={"step": step, best_key: val_metrics[best_key]},
                    )
                    print(f"[save] best checkpoint -> {out_dir}/scorer_best.pt")

    save_scorer(trainer.scorer, cfg, out_dir / "scorer.pt", extra={"step": global_step})
    print(f"[done] scorer -> {out_dir}/scorer.pt in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
