#!/usr/bin/env python3
"""Train the layer-1 pre-RoPE pseudo-token scorer (Qwen2.5-VL).

Run from the project root under the HAWK runtime::

    PYTHON_BIN="$(which python)" bash scripts/run.sh \
        scripts/train_layer1.py --out-dir outputs/lookhawk_layer1_smoke

The whole Qwen2.5-VL model is frozen and trimmed to (vision tower + token
embedding + decoder layers ``0 .. score_layer``); the only trainable state is
``P`` rows appended to the prompt embeddings. Each step runs two trimmed
forwards -- one with the learned rows (student), one with the real answer rows
(teacher) -- and matches their row-averaged per-head distributions over the
visual tokens. See ``src/lookhawk/layer1.py`` for the design and its trade-offs.

Optimization follows the LookaheadKV recipe: Adam (beta1 0.9, beta2 0.95),
cosine decay to ``--min-lr`` after a ``--warmup-ratio`` linear warmup, and
gradient clipping at 1. Dropout is off by construction -- the scorer is a bare
``nn.Parameter`` and the frozen decoder runs under ``eval()``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from lookhawk.layer1 import Layer1Trainer, save_scorer
from lookhawk.train import encode_jsonl, load_and_trim_model

# The evaluation harness runs Qwen2.5-VL at this floor (see
# scripts/evaluate_vlmeval.sh), so training encodes images the same way. Leaving
# it at the model default (3136) would train at V~500 and evaluate at V~1300 --
# a different visual-token count than the scorer was fitted on.
EVAL_MIN_PIXELS = 1280 * 28 * 28


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train-jsonl", default="data/smoke/train.jsonl")
    parser.add_argument("--val-jsonl", default="data/smoke/val.jsonl")
    parser.add_argument("--out-dir", default="outputs/lookhawk_layer1_smoke")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=0)

    # Scorer architecture
    parser.add_argument("--num-pseudo", type=int, default=32, help="appended rows P")
    parser.add_argument("--score-layer", type=int, default=1, help="decoder layer that scores")
    parser.add_argument(
        "--teacher-max-rows",
        type=int,
        default=32,
        help="max response length: how many answer tokens the teacher reads",
    )
    parser.add_argument("--temperature", type=float, default=1.0)

    # Data
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=EVAL_MIN_PIXELS,
        help=f"match the eval harness (default {EVAL_MIN_PIXELS}); 3136 for a fast smoke",
    )
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="processes for the CPU half of preprocessing (decode/resize/patchify)",
    )

    # Training
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-schedule", default="cosine", choices=["cosine", "none"])
    parser.add_argument("--warmup-ratio", type=float, default=0.02, help="fraction of steps")
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-best", action="store_true", help="save best-val-KL checkpoint")
    return parser


def _format_metrics(metrics: dict[str, float], keys: tuple[str, ...]) -> str:
    return " ".join(f"{key}={metrics[key]:.4f}" for key in keys if key in metrics)


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

    processor_kwargs = {"min_pixels": args.min_pixels}
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, **processor_kwargs
    )

    max_seq_len = args.max_seq_len or 2048
    print(f"[phase] encoding train rows (limit={args.limit_train})...", flush=True)
    train_samples, train_skips = encode_jsonl(
        processor,
        args.train_jsonl,
        max_seq_len=max_seq_len,
        limit=args.limit_train,
        workers=args.workers,
    )
    val_samples, val_skips = encode_jsonl(
        processor,
        args.val_jsonl,
        max_seq_len=max_seq_len,
        limit=args.limit_val,
        workers=args.workers,
    )
    print(
        f"[data] train={len(train_samples)} val={len(val_samples)} "
        f"(skips train={train_skips}, val={val_skips})"
    )
    if not train_samples:
        raise SystemExit("no training samples encoded; aborting")

    keep_layers = args.score_layer + 1
    t0 = time.time()
    print(f"[phase] loading & trimming model to {keep_layers} layer(s)...", flush=True)
    handle = load_and_trim_model(args.model_path, device, compute_dtype, keep_layers=keep_layers)
    print(f"[model] trimmed & loaded on {device} ({time.time() - t0:.1f}s)", flush=True)

    print("[phase] preparing per-sample prompt/answer caches...", flush=True)
    trainer = Layer1Trainer(
        handle,
        train_samples,
        device=device,
        compute_dtype=compute_dtype,
        num_pseudo=args.num_pseudo,
        score_layer=args.score_layer,
        teacher_max_rows=args.teacher_max_rows,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(
        f"[train] prepared {len(trainer.items)}/{len(train_samples)} samples on device",
        flush=True,
    )
    print(
        f"[cfg] num_pseudo={trainer.scorer.num_pseudo} score_layer={trainer.score_layer} "
        f"temperature={trainer.temperature} teacher_max_rows={trainer.teacher_max_rows} "
        f"min_pixels={args.min_pixels}",
        flush=True,
    )
    print(
        f"[opt] AdamW lr={args.lr} betas=(0.9, 0.95) weight_decay={args.weight_decay} "
        f"clip={args.max_grad_norm} schedule={args.lr_schedule} "
        f"warmup={args.warmup_ratio:.1%} min_lr={args.min_lr}",
        flush=True,
    )
    # A batch larger than the training split silently reuses samples within one
    # batch (``next_batch_ids`` wraps across epochs), so the gradient is over
    # fewer distinct samples than ``--batch-size`` claims.
    if len(trainer.items) < args.batch_size:
        print(
            f"[warn] batch_size={args.batch_size} exceeds the {len(trainer.items)} "
            f"usable training samples; some will repeat within a batch",
            flush=True,
        )

    # One-shot objective check. ``teacher_kl_uniform`` is the headroom the KL has
    # to work with: KL(teacher || uniform) over the visual tokens. If it sits near
    # zero the teacher is flat and this loss will report a number that barely
    # moves no matter how the pseudo rows are set.
    diag = trainer.metrics_for(trainer.items[0])
    print(
        "[diag] sample 0: kl={kl:.6f} teacher_kl_uniform={teacher_kl_uniform:.6f} ".format(**diag)
        + _format_metrics(diag, ("recall@1", "recall@16", "recall@64")),
        flush=True,
    )

    optimizer = trainer.build_optimizer(lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        trainer.build_scheduler(
            optimizer,
            total_steps=args.steps,
            warmup_ratio=args.warmup_ratio,
            min_lr=args.min_lr,
        )
        if args.lr_schedule == "cosine"
        else None
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    val_items: list = []
    for sample in val_samples:
        prepared = trainer._prepare(sample)
        if prepared is not None:
            val_items.append(prepared)
    print(f"[val] prepared {len(val_items)}/{len(val_samples)} val samples", flush=True)

    eval_keys = ("kl", "recall@1", "recall@2", "recall@16", "recall@64", "teacher_kl_uniform")
    best_val = float("inf")
    global_step = 0
    for step in range(1, args.steps + 1):
        batch_ids = trainer.next_batch_ids(args.batch_size)
        metrics = trainer.train_step(
            optimizer, batch_ids, max_grad_norm=args.max_grad_norm
        )
        if scheduler is not None:
            scheduler.step()
        global_step += 1
        if step == 1 or step % args.log_every == 0:
            # Wall-clock is part of the log because a batch of ``--batch-size`` is
            # accumulated one sample at a time, so the cost of a step scales with
            # it and the total run length is not obvious from ``--steps`` alone.
            print(
                f"[step {step}/{args.steps}] train_kl={metrics['train_kl']:.6f} "
                f"grad_norm={metrics['grad_norm']:.6f} lr={metrics['lr']:.3e} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        if args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            val_metrics = trainer.average_metrics(val_items)
            if val_metrics:
                print(f"[eval step {step}] {_format_metrics(val_metrics, eval_keys)}", flush=True)
                if args.save_best and val_metrics["kl"] < best_val:
                    best_val = val_metrics["kl"]
                    save_scorer(
                        trainer.scorer,
                        str(out_dir / "scorer_best.pt"),
                        extra={"step": step, "kl": val_metrics["kl"]},
                    )
                    print(f"[save] best checkpoint -> {out_dir}/scorer_best.pt", flush=True)

    save_scorer(trainer.scorer, str(out_dir / "scorer.pt"), extra={"step": global_step})
    print(f"[done] scorer -> {out_dir}/scorer.pt in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
