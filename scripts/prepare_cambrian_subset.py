#!/usr/bin/env python3
"""Turn a Cambrian-10M annotation dump into a train/val corpus, in two stages.

Stage ``filter`` -- the VisionSelector ``filter_json.py`` step. Streams the full
annotation once and routes each row to ``<dataset>_cambrian.jsonl`` by the
prefix carried on its ``image`` path. Whole corpus, no sampling, no split::

    {"id": 10095, "image": "chartqa/train/png/10095.png", "conversations": [...]}

Stage ``split`` -- carves a train/val subset out of one of those files (the
original annotation works too), **grouped by image**::

    {"image": "/abs/path/chartqa/train/png/10095.png",
     "question": "Is the value of Favorable 38 in 2015?", "answer": "Yes",
     "id": 10095}

Grouping is what makes the invariants below hold: the corpus carries several
annotation rows per image (ChartQA 1.54 rows/image, CoCo 4.11), and an earlier
version of this script sampled *rows*, which silently placed the same image on
both sides of the split.

Guarantees:

* ``--train-count`` / ``--val-count`` count **images**, drawn from one pool
  without replacement, so a val image can never appear in train. (Measured on
  the old row-based code: CoCo leaked 9.3% of its val images, ChartQA 1.4%.)
* ``--pairs-per-image`` draws that many q/a per image; at the default 1 the
  sample count equals the image count.

Nothing here is a defence against *benchmark* contamination -- the tarballs
this project pulls simply happen to ship train splits only (``chartqa/train/``,
``textvqa/train_images/``, ``coco/train2017/``). See README section 5.3.

This script only writes text: the jsonl pair and a newline-separated list of
archive members. It never touches the tarballs, so it is cheap to re-run. Feed
the member list to::

    tar -xzf datasets/<name>.tar.gz -C datasets -T <members file>

Why the q/a pairs are re-sampled rather than taken as-is: rows are multi-turn
and every row of an image shares that one picture, while the loader in
``lookhawk.data`` reads only the *last* pair of a row -- for this corpus the
least informative one. Drawing pairs at random per image keeps the whole pair
distribution in play. The draw is seeded, so re-running with a different
``--seed`` gives a fresh sample from the same extracted images at zero extra
image cost.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# The datasets this script knows how to select, keyed by the prefix their image
# paths carry inside the Cambrian annotation file.
DATASET_PREFIXES = {
    "ocr_vqa": "ocr_vqa/",
    "chartqa": "chartqa/",
    "textvqa": "textvqa/",
    "coco": "coco/",
}

# The image placeholder the chat template inserts on its own; the literal string
# left in the corpus text would otherwise be tokenized as real text.
_IMAGE_TAG = re.compile(r"<image>\s*")


def qa_pairs(conversations: list[dict]) -> list[tuple[str, str]]:
    """Pair each human turn with the assistant turn that follows it."""
    pairs: list[tuple[str, str]] = []
    pending: str | None = None
    for turn in conversations:
        role = str(turn.get("from", "")).lower()
        value = turn.get("value")
        if not isinstance(value, str):
            continue
        if role in {"human", "user"}:
            pending = value
        elif role in {"gpt", "assistant"} and pending is not None:
            pairs.append((pending, value))
            pending = None
    return pairs


def clean(text: str) -> str:
    return _IMAGE_TAG.sub("", text).strip()


def source_rows(path: Path, prefix: str):
    """Yield every annotation row whose ``image`` starts with ``prefix``."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            image = row.get("image")
            if isinstance(image, str) and image.startswith(prefix):
                yield row


def sample_images(path: Path, prefix: str, count: int,
                  rng: random.Random) -> tuple[list[str], int]:
    """One pass: uniform sample of ``count`` *distinct* image paths.

    Rows repeat images, so the reservoir runs over first sightings rather than
    over rows -- that is what keeps the draw free of duplicates.
    """
    seen: set[str] = set()
    kept: list[str] = []
    found = 0
    for row in source_rows(path, prefix):
        image = row["image"]
        if image in seen:
            continue
        seen.add(image)
        if found < count:
            kept.append(image)
        else:
            swap = rng.randint(0, found)
            if swap < count:
                kept[swap] = image
        found += 1
    return kept, found


def collect_pairs(path: Path, prefix: str, wanted: set[str],
                  ) -> dict[str, list[tuple[str, str, object]]]:
    """Second pass: pool the q/a pairs of every wanted image across its rows."""
    pooled: dict[str, list[tuple[str, str, object]]] = {}
    for row in source_rows(path, prefix):
        image = row["image"]
        if image not in wanted:
            continue
        row_id = row.get("id")
        pooled.setdefault(image, []).extend(
            (question, answer, row_id)
            for question, answer in qa_pairs(row.get("conversations") or [])
        )
    return pooled


def build_rows(images: list[str], pooled: dict, pairs_per_image: int,
               rng: random.Random, image_root: Path) -> list[dict]:
    out: list[dict] = []
    for image in images:
        pairs = pooled.get(image) or []
        if not pairs:
            continue
        chosen = rng.sample(pairs, min(pairs_per_image, len(pairs)))
        for question, answer, row_id in chosen:
            question, answer = clean(question), clean(answer)
            if not question or not answer:
                continue
            out.append(
                {
                    "image": str(image_root / image),
                    "question": question,
                    "answer": answer,
                    "id": row_id,
                }
            )
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_filter(args: argparse.Namespace) -> None:
    """Stage 1: split the annotation into one jsonl per dataset, verbatim."""
    source = Path(args.cambrian_jsonl)
    if not source.is_file():
        raise SystemExit(f"annotation file not found: {source}")
    out_dir = Path(args.out_dir or "datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    handles = {
        name: (out_dir / f"{name}_cambrian.jsonl").open("w", encoding="utf-8")
        for name in args.dataset
    }
    counts = {name: 0 for name in args.dataset}
    total = 0
    try:
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                image = row.get("image")
                if not isinstance(image, str):
                    continue
                total += 1
                for name in args.dataset:
                    if image.startswith(DATASET_PREFIXES[name]):
                        handles[name].write(
                            json.dumps(
                                {
                                    "id": row.get("id"),
                                    "image": image,
                                    "conversations": row.get("conversations"),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        counts[name] += 1
                        break
    finally:
        for handle in handles.values():
            handle.close()

    print(f"[read] {source.name}: {total} rows")
    for name in args.dataset:
        print(f"[write] {out_dir / f'{name}_cambrian.jsonl'}  ({counts[name]} rows)")


def run_split(args: argparse.Namespace) -> None:
    """Stage 2: group by image, then draw disjoint train/val image sets."""
    if len(args.dataset) != 1:
        raise SystemExit("--stage split takes exactly one --dataset")
    name = args.dataset[0]
    source = Path(args.cambrian_jsonl)
    if not source.is_file():
        raise SystemExit(f"annotation file not found: {source}")

    prefix = DATASET_PREFIXES[name]
    rng = random.Random(args.seed)
    wanted = args.train_count + args.val_count

    images, found = sample_images(source, prefix, wanted, rng)
    if len(images) < wanted:
        raise SystemExit(
            f"only {found} distinct {name} images available, need {wanted} "
            f"({args.train_count} train + {args.val_count} val).\n"
            f"Lower --train-count/--val-count; the pool is {found} images."
        )
    print(f"[read] {source.name}: {found} distinct {name} images, drew {len(images)}")

    # Shuffle before cutting so val is not biased towards the file head.
    rng.shuffle(images)
    val_images = images[: args.val_count]
    train_images = images[args.val_count :]
    overlap = set(val_images) & set(train_images)
    if overlap:
        raise SystemExit(f"internal error: {len(overlap)} images leaked across the split")
    print(f"[split] train={len(train_images)} images  val={len(val_images)} images  "
          f"overlap=0")

    image_root = Path(args.image_root).resolve()
    pooled = collect_pairs(source, prefix, set(images))
    no_pairs = [img for img in images if not pooled.get(img)]
    if no_pairs:
        print(f"[warn] {len(no_pairs)} images carry no usable q/a; they are dropped")

    val = build_rows(val_images, pooled, args.pairs_per_image, rng, image_root)
    train = build_rows(train_images, pooled, args.pairs_per_image, rng, image_root)
    print(f"[pair] {args.pairs_per_image} pair(s)/image -> "
          f"train={len(train)} samples, val={len(val)} samples")

    members = sorted({r["image"] for r in train + val})
    missing = [m for m in members if not Path(m).is_file()]
    print(f"[images] {len(members)} distinct needed, {len(missing)} not on disk yet")

    if args.dry_run:
        print("[dry-run] nothing written")
        return

    out_dir = Path(args.out_dir or f"data/cambrian-{name}")
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    members_file = Path(args.members_file or (out_dir / "members.txt"))
    members_file.parent.mkdir(parents=True, exist_ok=True)
    # tar -T wants member names exactly as they appear in the archive, which is
    # the annotation's own path relative to the extracted ``--image-root``.
    with members_file.open("w", encoding="utf-8") as handle:
        for member in members:
            handle.write(str(Path(member).relative_to(image_root)) + "\n")
    print(f"[write] {out_dir}/train.jsonl  {out_dir}/val.jsonl")
    print(f"[write] {members_file} ({len(members)} members)")
    if missing:
        archive = image_root / f"{name}.tar.gz"
        print(f"[extract] {len(missing)} images still missing; run:\n"
              f"  tar -xzf {archive} -C {image_root} -T {members_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=("filter", "split"), default="split",
                        help="filter: full per-dataset jsonl; split: train/val subset")
    parser.add_argument("--cambrian-jsonl", default="datasets/jsons/Cambrian737k.jsonl",
                        help="the annotation dump, or a <name>_cambrian.jsonl from --stage filter")
    parser.add_argument("--dataset", nargs="+", default=["ocr_vqa"],
                        choices=sorted(DATASET_PREFIXES),
                        help="filter: any number; split: exactly one")
    parser.add_argument("--image-root", default="datasets",
                        help="prefix for image paths, and the tar -C target")
    parser.add_argument("--out-dir", default=None,
                        help="filter defaults to datasets/, split to data/cambrian-<name>")
    parser.add_argument("--train-count", type=int, default=5000,
                        help="IMAGES in train (not rows, not samples)")
    parser.add_argument("--val-count", type=int, default=500,
                        help="IMAGES in val; drawn disjoint from train")
    parser.add_argument("--pairs-per-image", type=int, default=1,
                        help="q/a pairs kept per image; samples == images at 1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--members-file", default=None,
                        help="where to write the archive member list for tar -T")
    parser.add_argument("--dry-run", action="store_true",
                        help="report counts without writing anything")
    args = parser.parse_args()

    if args.stage == "filter":
        run_filter(args)
    else:
        run_split(args)


if __name__ == "__main__":
    sys.exit(main())
