#!/usr/bin/env python3
"""Control experiment for stage B of diagnose_lookhawk.py.

Stage B found that a given lookahead row's read output is ~0.97 cosine-similar
across samples -- i.e. barely sample-dependent. Before blaming the read
architecture, two confounds have to be ruled out:

1. Maybe *any* set of layer-0 outputs is ~0.97 similar, because Qwen's residual
   stream has a large common component. Control: read the REAL answer-token rows
   through the very same ``conditioned_layer0_read``. If their cross-sample
   similarity is also ~0.97, cosine on raw outputs is simply the wrong ruler.

2. Maybe ``embed_lookahead`` itself already collapsed, because it was trained
   under the old (provably degenerate) objective. Control: measure the input
   embeddings directly, no model forward.

Both controls are reported twice: raw, and after subtracting the global mean
direction over all (sample, row) pairs. The centred numbers are the ones that
decide the design -- they show whether a *sample-specific* component exists at
all, and how large it is relative to the shared one.

Read-only: loads the model, runs forward passes over decoder layer 0 only.

    PYTHON_BIN=/path/to/python bash scripts/run.sh \
        scripts/diagnose_read_control.py --device cuda:0
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from lookhawk.config import LookhawkConfig
from lookhawk.model_graph import IMAGE_TOKEN_ID, resolve_handle, verify_model_graph
from lookhawk.modules import Layer0Scorer
from lookhawk.read import build_prompt_embeds, conditioned_layer0_read
from lookhawk.train import encode_jsonl


# ---------------------------------------------------------------- statistics

def _cos_upper(rows: torch.Tensor) -> torch.Tensor:
    """Upper-triangle pairwise cosines of [N, D]."""
    unit = F.normalize(rows.float(), p=2, dim=-1)
    gram = unit @ unit.t()
    n = gram.shape[0]
    idx = torch.triu_indices(n, n, offset=1, device=gram.device)
    return gram[idx[0], idx[1]]


def _within(stack: torch.Tensor) -> list[float]:
    """Pairwise cosines among the rows *inside* each sample."""
    return [float(v) for s in stack for v in _cos_upper(s).tolist()]


def _across(stack: torch.Tensor) -> list[float]:
    """Cosines of the same row index *between* samples."""
    n, rows, _ = stack.shape
    out: list[float] = []
    for i in range(rows):
        out.extend(float(v) for v in _cos_upper(stack[:, i, :]).tolist())
    return out


def _effective_rank(flat: torch.Tensor) -> tuple[float, float]:
    """(top-1 energy share, participation ratio) of the centred matrix."""
    centred = flat.float() - flat.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centred)
    energy = singular.pow(2)
    share = energy / energy.sum().clamp_min(1e-30)
    return float(share[0].item()), float(1.0 / share.pow(2).sum().clamp_min(1e-30).item())


def _summarise(name: str, stack: torch.Tensor) -> dict:
    """Raw vs mean-centred within/across similarity for an [N, L, D] stack."""
    n, rows, dim = stack.shape
    flat = stack.reshape(n * rows, dim).float()
    mean_dir = flat.mean(dim=0, keepdim=True)
    centred = (flat - mean_dir).reshape(n, rows, dim)

    # How much of a row is just the shared direction: ||mean|| / mean(||row||).
    shared_ratio = float(mean_dir.norm().item() / flat.norm(dim=-1).mean().clamp_min(1e-9).item())

    raw_within, raw_across = _within(stack), _across(stack)
    cen_within, cen_across = _within(centred), _across(centred)
    top1, pr = _effective_rank(flat)
    return {
        "name": name,
        "shared_ratio": shared_ratio,
        "raw_within": raw_within,
        "raw_across": raw_across,
        "cen_within": cen_within,
        "cen_across": cen_across,
        "top1": top1,
        "pr": pr,
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _fmt(xs: list[float]) -> str:
    return f"mean={_mean(xs):+.4f} min={min(xs):+.4f} max={max(xs):+.4f}"


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
    p.add_argument("--random-seed", type=int, default=0,
                   help="seed for the untrained-row control")
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

    checkpoint = torch.load(args.scorer, map_location="cpu")
    cfg = LookhawkConfig(**checkpoint["config"])
    print(f"[config] lookahead_size={cfg.lookahead_size} teacher_max_rows={cfg.teacher_max_rows} "
          f"lora_rank={cfg.lora_rank} lora_targets={cfg.lora_targets} loss={cfg.loss} "
          f"score_lora_targets={cfg.score_lora_targets}")

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    handle = resolve_handle(model)
    verify_model_graph(handle)
    layer0 = handle.layer0()

    scorer = Layer0Scorer.from_layer0(layer0, cfg).to(device=device, dtype=dtype).eval()
    missing, _ = scorer.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing:
        print(f"[scorer] missing keys kept at init: {sorted(missing)}")

    # The smoke answers are only 4-6 tokens, so both blocks are cut to the
    # shortest answer present instead of the configured lookahead_size.
    lengths = [int(s.answer_ids.numel()) for s in samples]
    rows_per_sample = min(int(cfg.lookahead_size), min(lengths))
    if rows_per_sample < 2:
        raise SystemExit(f"shortest answer is {min(lengths)} tokens; need at least 2 rows")
    print(f"[rows] answer lengths {sorted(lengths)} -> using {rows_per_sample} rows per block")

    trained_rows = scorer.embed_lookahead.detach()[:rows_per_sample].float()
    trained_norms = trained_rows.norm(dim=-1, keepdim=True)
    print(f"[scorer] trained row norms {[round(v, 4) for v in trained_norms.flatten().tolist()]}")

    generator = torch.Generator(device="cpu").manual_seed(args.random_seed)
    native_random = torch.randn(rows_per_sample, trained_rows.shape[1], generator=generator) * cfg.embed_scale
    # Random *directions* rescaled to the trained rows' norms: isolates direction
    # from magnitude, so a difference cannot be blamed on the rows being small.
    matched_random = F.normalize(
        torch.randn(rows_per_sample, trained_rows.shape[1], generator=generator).to(trained_norms.device),
        p=2, dim=-1,
    ) * trained_norms
    print(f"[control] native random row norms "
          f"{[round(v, 4) for v in native_random.norm(dim=-1).tolist()]}")
    stacks: dict[str, list[torch.Tensor]] = {
        "learned_embed": [],   # the raw learnable rows (no forward at all)
        "teacher_embed": [],   # real answer-token embeddings, same row count
        "learned_read": [],    # learned rows after conditioned_layer0_read, no LoRA
        "teacher_read": [],    # answer rows after the SAME read -- the key control
        "learned_lora_read": [],  # the real training/inference path (full rows + LoRA)
        "random_native_read": [],   # UNTRAINED rows at the training init scale
        "random_matched_read": [],  # random directions at the trained rows' norms
    }

    for sample in samples:
        prompt_ids = sample.prompt_ids.to(device)
        visual_indices = torch.where(prompt_ids[0] == IMAGE_TOKEN_ID)[0].long()
        if visual_indices.numel() == 0:
            continue
        prompt_embeds = build_prompt_embeds(
            handle, prompt_ids,
            pixel_values=sample.pixel_values.to(device) if sample.pixel_values is not None else None,
            image_grid_thw=sample.image_grid_thw.to(device) if sample.image_grid_thw is not None else None,
        )
        answer_ids = sample.answer_ids[0].to(device)
        if answer_ids.numel() < rows_per_sample:
            continue
        answer_slice = answer_ids[:rows_per_sample]

        # Cut to the same row count as the answer block, otherwise the read below
        # would treat the leftover learned rows as prompt prefix.
        learned = scorer.embed_lookahead[:rows_per_sample].unsqueeze(0).to(dtype=prompt_embeds.dtype)
        teacher = handle.embed_tokens()(answer_slice).unsqueeze(0).to(dtype=prompt_embeds.dtype)
        stacks["learned_embed"].append(learned[0].float().cpu())
        stacks["teacher_embed"].append(teacher[0].float().cpu())

        with torch.no_grad():
            for key, block in (
                ("learned_read", learned),
                ("teacher_read", teacher),
                ("random_native_read",
                 native_random.unsqueeze(0).to(dtype=prompt_embeds.dtype, device=device)),
                ("random_matched_read",
                 matched_random.unsqueeze(0).to(dtype=prompt_embeds.dtype, device=device)),
            ):
                seq = torch.cat([prompt_embeds, block], dim=1)
                # scorer=None on purpose: isolate the read architecture, not the LoRA.
                out = conditioned_layer0_read(layer0, seq, appended_size=rows_per_sample, scorer=None)
                stacks[key].append(out[0].float().cpu())
            # Reproduce the real path stage B measured: all configured rows + LoRA.
            full_rows = scorer.embed_lookahead.unsqueeze(0).to(dtype=prompt_embeds.dtype)
            seq = torch.cat([prompt_embeds, full_rows], dim=1)
            out = conditioned_layer0_read(layer0, seq, appended_size=int(full_rows.shape[1]), scorer=scorer)
            stacks["learned_lora_read"].append(out[0].float().cpu())
        if device.type == "cuda":
            torch.cuda.empty_cache()

    usable = len(stacks["learned_read"])
    if usable < 2:
        raise SystemExit("need at least two usable samples")
    print(f"\n================ usable samples={usable} rows_per_sample={rows_per_sample}")

    summaries = []
    for name, per_sample in stacks.items():
        stacked = torch.stack(per_sample).float()
        summaries.append(_summarise(name, stacked))

    def width(xs: list[float]) -> float:
        return max(xs) - min(xs)

    print("\n--- raw cosine (the ruler stage B used) ---")
    for s in summaries:
        print(f"  {s['name']:<14s} within={_mean(s['raw_within']):+.4f} "
              f"across={_mean(s['raw_across']):+.4f} "
              f"gap={_mean(s['raw_within']) - _mean(s['raw_across']):+.4f}")

    print("\n--- AFTER subtracting the global mean direction ---")
    print("    (this is whether a sample-specific component exists at all)")
    for s in summaries:
        print(f"  {s['name']:<14s} within={_mean(s['cen_within']):+.4f} "
              f"across={_mean(s['cen_across']):+.4f} "
              f"gap={_mean(s['cen_within']) - _mean(s['cen_across']):+.4f} "
              f"| across min={min(s['cen_across']):+.4f} max={max(s['cen_across']):+.4f}")

    print("\n--- shared direction & dimensionality ---")
    for s in summaries:
        print(f"  {s['name']:<14s} ||global mean|| / mean||row|| = {s['shared_ratio']:.4f} "
              f"| top-1 energy share = {s['top1']:.4f} | participation ratio = {s['pr']:.2f} dims")

    print("\n[how to read]")
    print("  * If teacher_read has the SAME raw cross-sample cos as learned_read, then")
    print("    stage B's raw number says nothing -- layer-0 outputs simply share a big")
    print("    common direction. Compare the CENTRED block instead.")
    print("  * participation ratio near 1 => every row across every sample lies on one")
    print("    line => no sample-dependent information survives the read.")


if __name__ == "__main__":
    main()
