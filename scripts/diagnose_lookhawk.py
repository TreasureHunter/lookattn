#!/usr/bin/env python3
"""Offline diagnostic for the proposed layer-1 attention-KL design.

Answers three questions with numbers, before any training code is written:

A. Is the near-uniform HAWK score an artifact of the per-head L2 normalization,
   or is each head's raw score vector already flat, or is the flatness produced
   by summing 28 heads?  Reported per head (raw spread, L2-normalised spread,
   softmax peakedness at several temperatures) plus the aggregated vector.

B. Are the 32 read-output rows distinguishable within a sample, and -- the
   question that decides the whole design -- how similar is a given row's
   read output *across* samples?  If cross-sample queries are near-identical,
   the conditioning is dead no matter which layer supplies the projections.

C. How peaked is the model's own attention over visual tokens at layers 0/1/2,
   and what fraction of the query's attention mass lands on visual tokens at
   all?  If the visual segment is a thin, flat tail, a KL defined only on
   visual tokens has no gradient (this is the risk flagged for that choice).

Read-only: loads the model, runs forward passes, writes nothing.

    PYTHON_BIN="$(which python)" bash scripts/run.sh \
        python scripts/diagnose_lookhawk.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from hawk.pruning import repeat_kv
from lookhawk.config import LookhawkConfig
from lookhawk.model_graph import IMAGE_TOKEN_ID, resolve_handle, verify_model_graph
from lookhawk.modules import Layer0Scorer
from lookhawk.read import (
    build_prompt_embeds,
    conditioned_layer0_read,
    select_teacher_answer_ids,
)
from lookhawk.train import default_head_weights, encode_jsonl


# ---------------------------------------------------------------- helpers

def _cos_matrix(rows: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine of [N, D] rows, returned as the upper triangle (N*(N-1)/2,)."""
    unit = F.normalize(rows.float(), p=2, dim=-1)
    gram = unit @ unit.t()
    n = gram.shape[0]
    idx = torch.triu_indices(n, n, offset=1, device=gram.device)
    # NOTE: gram[idx] with a [2, K] index tensor is *row-wise* advanced indexing
    # and yields [2, K, n]; the coordinate form is what returns [K].
    return gram[idx[0], idx[1]]


def _peakedness(dist: torch.Tensor) -> dict[str, float]:
    """Entropy ratio / max weight / top-5% mass of a distribution over the last dim."""
    v = dist.shape[-1]
    p = dist.float()
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(p * (p + 1e-12).log()).sum(dim=-1)
    top_count = max(1, int(round(0.05 * v)))
    top_mass = p.topk(top_count, dim=-1).values.sum(dim=-1)
    return {
        "entropy_ratio": float((entropy / math.log(v)).mean().item()),
        "max_weight": float(p.max(dim=-1).values.mean().item()),
        "top5_mass": float(top_mass.mean().item()),
        "uniform_max": 1.0 / v,
        "uniform_top5": top_count / v,
    }


def _softmax_peakedness(scores: torch.Tensor, temps: tuple[float, ...]) -> dict[float, dict[str, float]]:
    return {t: _peakedness(F.softmax(scores.float() / t, dim=-1)) for t in temps}


# ---------------------------------------------------------------- stage A

def stage_a(layer0, prompt_embeds, visual_indices, query_embeds) -> dict:
    """Per-head raw / L2-normalised / aggregated scores from the layer-0 HAWK path."""
    attn = layer0.self_attn
    heads, head_dim = int(attn.num_heads), int(attn.head_dim)

    q = attn.q_proj(layer0.input_layernorm(query_embeds))
    b, q_len, _ = q.shape
    q = q.view(b, q_len, heads, head_dim).transpose(1, 2)

    k = attn.k_proj(layer0.input_layernorm(prompt_embeds))
    kv_len = k.shape[1]
    k = k.view(b, kv_len, int(attn.num_key_value_heads), head_dim).transpose(1, 2)
    k = repeat_kv(k, int(attn.num_key_value_groups))
    visual_keys = k.index_select(2, visual_indices)

    raw = torch.matmul(q, visual_keys.transpose(2, 3)) / math.sqrt(head_dim)
    raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).mean(dim=2).squeeze(0)

    normed = F.normalize(raw, p=2, dim=-1).float()
    return {"raw": raw.float(), "normed": normed, "heads": heads}


# ---------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="models/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--jsonl", nargs="+",
                   default=["data/smoke/train.jsonl", "data/smoke/val.jsonl",
                            "data/smoke/natural1.jsonl"])
    p.add_argument("--scorer", default="outputs/lookhawk_smoke_vision/scorer.pt")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn-layers", type=int, nargs="+", default=[0, 1, 2],
                   help="decoder layers whose real attention over visual tokens is measured")
    p.add_argument("--out", default="outputs/diagnose_lookhawk.json",
                   help="path for the raw aggregate dump (so a later crash costs nothing)")
    return p.parse_args()


def main() -> None:
    args = build_parser()
    from transformers import AutoProcessor, AutoModelForVision2Seq

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    samples = []
    for path in args.jsonl:
        got, skips = encode_jsonl(processor, path, max_seq_len=args.max_seq_len, limit=args.limit)
        print(f"[data] {path}: {len(got)} samples (skips={skips})")
        samples.extend(got)
    if not samples:
        raise SystemExit("no samples encoded")

    cfg = LookhawkConfig()
    checkpoint = None
    if args.scorer and Path(args.scorer).is_file():
        checkpoint = torch.load(args.scorer, map_location="cpu")
        cfg = LookhawkConfig(**checkpoint["config"])
        print(f"[scorer] {args.scorer}")
    else:
        print("[scorer] none found -> random-init embed_lookahead, no LoRA weights")

    t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    handle = resolve_handle(model)
    info = verify_model_graph(handle)
    print(f"[model] loaded on {device}: {info['num_layers']} layers, "
          f"{info['num_heads']} heads, head_dim={info['head_dim']}")
    layer0 = handle.layer0()
    head_weights = default_head_weights().to(device)

    scorer = Layer0Scorer.from_layer0(layer0, cfg).to(device=device, dtype=dtype).eval()
    if checkpoint is not None:
        # Older checkpoints predate the score-stage LoRA, so load non-strictly and
        # report what did not line up rather than aborting the diagnostic.
        missing, unexpected = scorer.load_state_dict(checkpoint["state_dict"], strict=False)
        if missing:
            print(f"[scorer] missing keys kept at init: {sorted(missing)}")
        if unexpected:
            print(f"[scorer] unexpected keys ignored: {sorted(unexpected)}")

    # ------------------------------------------------------------ collect
    per_head_raw_spread: list[float] = []
    per_head_norm_spread: list[float] = []
    agg_spread: list[float] = []
    softmax_by_temp: dict[float, list[dict[str, float]]] = {t: [] for t in (1.0, 0.1, 0.01, 1e-3)}
    agg_softmax: dict[float, list[dict[str, float]]] = {t: [] for t in (1.0, 0.1, 0.01, 1e-3)}

    within_row_cos: list[float] = []
    cross_row_cos: dict[int, list[float]] = {}
    read_vs_embed_cos: list[float] = []
    read_delta_ratio: list[float] = []

    attn_layer_stats: dict[int, list[dict[str, float]]] = {i: [] for i in args.attn_layers}
    attn_layer_stats_avg: dict[int, list[dict[str, float]]] = {i: [] for i in args.attn_layers}
    visual_mass: dict[int, list[float]] = {i: [] for i in args.attn_layers}

    visual_counts: list[int] = []
    kept_rows: list[torch.Tensor] = []  # per-sample [R, hidden] read outputs

    for index, sample in enumerate(samples):
        prompt_ids = sample.prompt_ids.to(device)
        visual_indices = torch.where(prompt_ids[0] == IMAGE_TOKEN_ID)[0].long()
        if visual_indices.numel() == 0:
            continue
        visual_counts.append(int(visual_indices.numel()))

        prompt_embeds = build_prompt_embeds(
            handle, prompt_ids,
            pixel_values=sample.pixel_values.to(device) if sample.pixel_values is not None else None,
            image_grid_thw=sample.image_grid_thw.to(device) if sample.image_grid_thw is not None else None,
        )

        answer_ids = sample.answer_ids[0].to(device)
        teacher_ids = select_teacher_answer_ids(answer_ids, cfg)
        if teacher_ids.numel() == 0:
            continue
        answer_embeds = handle.embed_tokens()(teacher_ids).unsqueeze(0)

        # ---- A: per-head spread, teacher query (answer rows) -----------------
        with torch.no_grad():
            a = stage_a(layer0, prompt_embeds, visual_indices, answer_embeds)
        raw, normed = a["raw"], a["normed"]                       # [H, V]
        per_head_raw_spread.append(float((raw.std(dim=-1) / raw.abs().mean(dim=-1).clamp_min(1e-9)).mean()))
        per_head_norm_spread.append(float((normed.std(dim=-1) / normed.abs().mean(dim=-1).clamp_min(1e-9)).mean()))
        aggregated = (normed * head_weights[:, None]).sum(dim=0)
        agg_spread.append(float(aggregated.std() / aggregated.abs().mean().clamp_min(1e-9)))
        for t in softmax_by_temp:
            softmax_by_temp[t].append(_peakedness(F.softmax(normed / t, dim=-1)))
            agg_softmax[t].append(_peakedness(F.softmax(aggregated / t, dim=-1)))

        # ---- B: read-output rows --------------------------------------------
        rows = scorer.embed_lookahead.unsqueeze(0)
        seq = torch.cat([prompt_embeds, rows.to(dtype=prompt_embeds.dtype)], dim=1)
        with torch.no_grad():
            h = conditioned_layer0_read(layer0, seq, appended_size=int(rows.shape[1]), scorer=scorer)[0]
        kept_rows.append(h.float().cpu())
        within_row_cos.extend(_cos_matrix(h).tolist())
        read_vs_embed_cos.extend(
            F.cosine_similarity(h.float(), scorer.embed_lookahead.float(), dim=-1).tolist()
        )
        delta = (h.float() - scorer.embed_lookahead.float()).norm(dim=-1)
        base = scorer.embed_lookahead.float().norm(dim=-1).clamp_min(1e-9)
        read_delta_ratio.extend((delta / base).tolist())

        # ---- C: the model's own attention over visual tokens -----------------
        full_ids = torch.cat([prompt_ids, sample.answer_ids.to(device)], dim=1)
        answer_positions = torch.arange(prompt_ids.shape[1], full_ids.shape[1], device=device)
        with torch.no_grad():
            out = model(
                input_ids=full_ids,
                pixel_values=sample.pixel_values.to(device) if sample.pixel_values is not None else None,
                image_grid_thw=sample.image_grid_thw.to(device) if sample.image_grid_thw is not None else None,
                output_attentions=True,
                use_cache=False,
            )
        for layer_index in args.attn_layers:
            if layer_index >= len(out.attentions):
                continue
            attn = out.attentions[layer_index][0].float()          # [H, T, T]
            rows_sel = attn.index_select(1, answer_positions)      # [H, R, T]
            over_visual = rows_sel[..., visual_indices]            # [H, R, V]
            mass = over_visual.sum(dim=-1).mean().item()           # visual share of the row
            visual_mass[layer_index].append(float(mass))
            # Per (head, answer-row) distribution over visual tokens: this is the
            # teacher distribution a KL *restricted to visual tokens* would use,
            # so its peakedness is exactly what decides whether that KL has
            # gradient. Averaging rows first would smooth the peaks away.
            flat = over_visual.reshape(-1, over_visual.shape[-1])
            attn_layer_stats[layer_index].append(_peakedness(flat))
            attn_layer_stats_avg[layer_index].append(
                _peakedness(over_visual.mean(dim=1))
            )
        del out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not visual_counts:
        raise SystemExit("no usable samples")

    rows_per_sample = kept_rows[0].shape[0]
    for i in range(rows_per_sample):
        stack = torch.stack([r[i] for r in kept_rows])             # [N, hidden]
        cross_row_cos[i] = _cos_matrix(stack).tolist()

    # ------------------------------------------------------------ report
    def mean(xs: list[float]) -> float:
        return sum(xs) / max(1, len(xs))

    def fmt(xs: list[float]) -> str:
        return f"mean={mean(xs):+.4f} min={min(xs):+.4f} max={max(xs):+.4f}"

    print(f"\n================ samples={len(visual_counts)} "
          f"avg_visual_tokens={mean([float(v) for v in visual_counts]):.1f}")

    print("\n--- A. per-head score spread (layer-0 HAWK path, teacher query) ---")
    print(f"  raw per-head      std/|mean| = {mean(per_head_raw_spread):.4f}")
    print(f"  L2-normed per-head std/|mean| = {mean(per_head_norm_spread):.4f}")
    print(f"  aggregated (Σ w_h)  std/|mean| = {mean(agg_spread):.4f}")
    print("  softmax peakedness of the AGGREGATED vector (entropy_ratio 1.0 == uniform):")
    for t in agg_softmax:
        s = agg_softmax[t]
        print(f"    τ={t:<7g} entropy_ratio={mean([d['entropy_ratio'] for d in s]):.5f} "
              f"max={mean([d['max_weight'] for d in s]):.2e} "
              f"top5%={mean([d['top5_mass'] for d in s]):.4f} "
              f"(uniform max={s[0]['uniform_max']:.2e}, top5%={s[0]['uniform_top5']:.4f})")
    print("  softmax peakedness PER HEAD:")
    for t in softmax_by_temp:
        s = softmax_by_temp[t]
        print(f"    τ={t:<7g} entropy_ratio={mean([d['entropy_ratio'] for d in s]):.5f} "
              f"max={mean([d['max_weight'] for d in s]):.2e} "
              f"top5%={mean([d['top5_mass'] for d in s]):.4f}")

    print("\n--- B. read-output rows ---")
    print(f"  within-sample  pairwise cos among {rows_per_sample} rows : {fmt(within_row_cos)}")
    all_cross = [v for vals in cross_row_cos.values() for v in vals]
    print(f"  ACROSS-sample  same-row-index cos (row 0 vs row 0, ...)  : {fmt(all_cross)}")
    print(f"  cos(read row, its own embed_lookahead)                   : {fmt(read_vs_embed_cos)}")
    print(f"  ||read - embed|| / ||embed||  (sample-dependent share)   : {fmt(read_delta_ratio)}")

    print("\n--- C. the model's own attention over visual tokens ---")
    for layer_index in args.attn_layers:
        stats = attn_layer_stats[layer_index]
        avg_stats = attn_layer_stats_avg[layer_index]
        if not stats:
            continue
        print(f"  layer {layer_index}: visual share of the answer row's attention mass = "
              f"{mean(visual_mass[layer_index]):.4f}")
        print(f"           PER (head, answer row) over visual tokens: entropy_ratio="
              f"{mean([d['entropy_ratio'] for d in stats]):.5f} "
              f"max={mean([d['max_weight'] for d in stats]):.2e} "
              f"top5%={mean([d['top5_mass'] for d in stats]):.4f} "
              f"(uniform max={stats[0]['uniform_max']:.2e}, top5%={stats[0]['uniform_top5']:.4f})")
        print(f"           rows-averaged first (28 heads):            entropy_ratio="
              f"{mean([d['entropy_ratio'] for d in avg_stats]):.5f} "
              f"max={mean([d['max_weight'] for d in avg_stats]):.2e} "
              f"top5%={mean([d['top5_mass'] for d in avg_stats]):.4f}")

    print("\n[read] entropy_ratio near 1.0 => the distribution is flat => KL has ~no gradient.")

    if args.out:
        payload = {
            "samples": len(visual_counts),
            "avg_visual_tokens": mean([float(v) for v in visual_counts]),
            "a": {
                "per_head_raw_spread": mean(per_head_raw_spread),
                "per_head_norm_spread": mean(per_head_norm_spread),
                "agg_spread": mean(agg_spread),
                "softmax_by_temp": {
                    str(t): {k: mean([d[k] for d in s]) for k in s[0]}
                    for t, s in softmax_by_temp.items()
                },
                "agg_softmax": {
                    str(t): {k: mean([d[k] for d in s]) for k in s[0]}
                    for t, s in agg_softmax.items()
                },
            },
            "b": {
                "rows_per_sample": rows_per_sample,
                "within_row_cos": {"mean": mean(within_row_cos),
                                   "min": min(within_row_cos),
                                   "max": max(within_row_cos)},
                "cross_row_cos": {"mean": mean(all_cross),
                                  "min": min(all_cross),
                                  "max": max(all_cross)},
                "read_vs_embed_cos": {"mean": mean(read_vs_embed_cos),
                                      "min": min(read_vs_embed_cos),
                                      "max": max(read_vs_embed_cos)},
                "read_delta_ratio": {"mean": mean(read_delta_ratio),
                                     "min": min(read_delta_ratio),
                                     "max": max(read_delta_ratio)},
            },
            "c": {
                str(layer_index): {
                    "visual_mass": mean(visual_mass[layer_index]),
                    "per_row": {k: mean([d[k] for d in attn_layer_stats[layer_index]])
                                for k in attn_layer_stats[layer_index][0]},
                    "rows_averaged": {k: mean([d[k] for d in attn_layer_stats_avg[layer_index]])
                                      for k in attn_layer_stats_avg[layer_index][0]},
                }
                for layer_index in args.attn_layers
                if attn_layer_stats[layer_index]
            },
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
