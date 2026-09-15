"""Format-agnostic image-QA jsonl adapter for lookahead training.

Each jsonl line is one of::

    {"image": <abs path>, "question": str, "answer": str}
    {"image": <abs path>, "conversations": [{"from": "human"|"gpt", "value": str}, ...]}
    {"image": <abs path>, "messages": [{"role": "user", "content": [...]}, ...]}

``encode_example`` builds the prompt-only and prompt+answer chat templates from
the *same* image, tokenizes both with ``truncation=False``, and asserts the
prompt ids are a prefix of the full ids (guaranteed because the image grid is
identical) so the answer span can be sliced safely. This is the safety net the
rest of the pipeline relies on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch

from .model_graph import IMAGE_TOKEN_ID, IM_END_TOKEN_ID

# Token ids stripped from the end of an assistant answer (chat-template tails).
_END_TURN_IDS = frozenset({IM_END_TOKEN_ID})
_NEWLINE_IDS = frozenset({198, 13})

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"})


class SkipRow(Exception):
    """Raised to skip a dataset row with an explainable reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class Sample:
    """One encoded image-QA example (all tensors CPU)."""

    prompt_ids: torch.Tensor  # [1, T] including image tokens
    answer_ids: torch.Tensor  # [1, K], K >= 1, stripped of chat tails
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    image_path: str

    @property
    def prompt_len(self) -> int:
        return int(self.prompt_ids.shape[1])

    @property
    def visual_token_count(self) -> int:
        return int((self.prompt_ids[0] == IMAGE_TOKEN_ID).sum().item())


def extract_qa(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return ``(image_path, question, answer)`` from any supported schema."""
    image = row.get("image")
    if not isinstance(image, str) or not image:
        raise SkipRow("missing image path")
    if not Path(image).suffix.lower() in _IMAGE_SUFFIXES:
        raise SkipRow(f"unsupported (non-image?) file: {image}")

    answer: str | None = None
    question: str | None = None

    if "answer" in row and "question" in row:
        question = row["question"]
        answer = row["answer"]
    elif "conversations" in row:
        turns = row["conversations"]
        for turn in reversed(turns):
            role = str(turn.get("from", "")).lower()
            if role in {"gpt", "assistant"} and not answer:
                answer = turn.get("value")
            if role in {"human", "user"} and question is None:
                question = turn.get("value")
        if question is None:  # keep the text of the last human turn found
            for turn in turns:
                role = str(turn.get("from", "")).lower()
                if role in {"human", "user"}:
                    question = turn.get("value")
    elif "messages" in row:
        messages = row["messages"]
        for message in reversed(messages):
            role = str(message.get("role", "")).lower()
            if role == "assistant" and not answer:
                answer = _content_text(message.get("content"))
            if role == "user" and question is None:
                question = _content_text(message.get("content"))
    else:
        raise SkipRow("no supported schema (need question/answer or conversations/messages)")

    if not isinstance(answer, str) or not answer.strip():
        raise SkipRow("empty answer")
    if not isinstance(question, str) or not question.strip():
        raise SkipRow("empty question")
    return image, question, answer


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return ""


def iter_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SkipRow(f"line {line_number}: invalid json ({exc})")


def _messages(image_path: str, question: str, answer: str | None) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [
        {"type": "image", "image": image_path},
        {"type": "text", "text": question},
    ]
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def _strip_answer_tail(ids: torch.Tensor) -> torch.Tensor:
    """Drop trailing chat-template tokens (``<|im_end|>`` and a final newline)."""
    while ids.numel() > 0 and int(ids[-1]) in _END_TURN_IDS:
        ids = ids[:-1]
    while ids.numel() > 0 and int(ids[-1]) in _NEWLINE_IDS:
        ids = ids[:-1]
    return ids


_PLACEHOLDER = "<|placeholder|>"


def _tokenize_with_image_grid(
    processor: Any, text: str, grid: torch.Tensor
) -> torch.Tensor:
    """Tokenize ``text``, expanding its one image placeholder from ``grid``.

    Mirrors what ``Qwen2_5_VLProcessor.__call__`` does after it has run the
    image processor: ``<|image_pad|>`` becomes ``prod(grid) // merge_size**2``
    copies of itself. Splitting the two apart is what lets ``encode_example``
    run the image processor once for both texts instead of once per text.
    """
    image_token = processor.image_token
    occurrences = text.count(image_token)
    if occurrences != 1:
        raise SkipRow(f"expected exactly one image placeholder, found {occurrences}")

    merge_length = int(processor.image_processor.merge_size) ** 2
    expanded = text.replace(image_token, _PLACEHOLDER * (int(grid.prod()) // merge_length))
    encoded = processor.tokenizer(
        [expanded.replace(_PLACEHOLDER, image_token)], return_tensors="pt"
    )
    return encoded["input_ids"][0]


def encode_example(
    processor: Any,
    image_path: str,
    question: str,
    answer: str,
    max_seq_len: int = 2048,
    image_token_id: int = IMAGE_TOKEN_ID,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> Sample | None:
    """Encode one example or return ``None`` when it must be skipped.

    ``processor`` is an ``AutoProcessor`` configured with the same
    ``min_pixels``/``max_pixels`` as inference.
    """
    if not Path(image_path).is_file():
        raise SkipRow(f"image file not found: {image_path}")

    prompt_messages = _messages(image_path, question, answer=None)
    full_messages = _messages(image_path, question, answer=answer)
    images, _ = _load_images(prompt_messages)

    text_prompt = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    text_full = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    # The image processor is roughly two thirds of the cost of a
    # ``processor(...)`` call and both texts share the identical image, so run
    # it once and tokenize each text against that one grid.
    image_inputs = processor.image_processor(images=images, return_tensors="pt")
    grid = image_inputs["image_grid_thw"]

    prompt_ids = _tokenize_with_image_grid(processor, text_prompt, grid[0])
    full_ids = _tokenize_with_image_grid(processor, text_full, grid[0])
    del min_pixels, max_pixels  # handled at processor construction

    prompt_len = prompt_ids.numel()
    if not torch.equal(prompt_ids, full_ids[:prompt_len]):
        raise SkipRow("prompt ids are not a prefix of full ids (template drift)")

    if prompt_len > max_seq_len:
        raise SkipRow(f"prompt too long ({prompt_len} > {max_seq_len})")

    answer_ids = full_ids[prompt_len:]
    answer_ids = _strip_answer_tail(answer_ids)
    remaining = max_seq_len - prompt_len
    if answer_ids.numel() == 0:
        raise SkipRow("answer empty after chat-tail strip")
    if remaining < 1:
        raise SkipRow("no room for any answer token")
    if answer_ids.numel() > remaining:
        answer_ids = answer_ids[:remaining]

    if not (prompt_ids == image_token_id).any():
        raise SkipRow("prompt contains no image tokens")

    return Sample(
        prompt_ids=prompt_ids.unsqueeze(0),
        answer_ids=answer_ids.unsqueeze(0),
        pixel_values=image_inputs.get("pixel_values"),
        image_grid_thw=image_inputs.get("image_grid_thw"),
        image_path=image_path,
    )


def _load_images(messages: list[dict[str, Any]]) -> tuple[Any, Any]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("qwen_vl_utils is required to load images") from exc

    images, videos = process_vision_info(messages)
    if not images:
        raise SkipRow("no image found by qwen_vl_utils")
    if videos:
        raise SkipRow("video inputs are not supported")
    return images, videos


def encode_rows(
    processor: Any,
    rows: Iterable[dict[str, Any]],
    max_seq_len: int = 2048,
    limit: int | None = None,
    skip_log: dict[str, int] | None = None,
    workers: int = 1,
) -> Iterator[Sample]:
    """Encode a stream of rows, skipping and counting reasons.

    ``workers > 1`` runs the encode in that many processes. Encoding is the CPU
    half of preprocessing (image decode, resize, patchify) and is independent
    per row, so it parallelizes cleanly; the vision tower that follows is not.
    """
    if workers <= 1:
        yield from _encode_rows_serial(processor, rows, max_seq_len, limit, skip_log)
        return
    yield from _encode_rows_parallel(
        processor, rows, max_seq_len, limit, skip_log, workers
    )


def _record_skip(skip_log: dict[str, int] | None, reason: str | None) -> None:
    if skip_log is not None and reason:
        skip_log[reason] = skip_log.get(reason, 0) + 1


def _encode_rows_serial(
    processor: Any,
    rows: Iterable[dict[str, Any]],
    max_seq_len: int,
    limit: int | None,
    skip_log: dict[str, int] | None,
) -> Iterator[Sample]:
    yielded = 0
    for row in rows:
        if limit is not None and yielded >= limit:
            break
        try:
            image_path, question, answer = extract_qa(row)
            sample = encode_example(
                processor, image_path, question, answer, max_seq_len=max_seq_len
            )
        except SkipRow as exc:
            _record_skip(skip_log, exc.reason)
            continue
        if sample is None:
            continue
        yielded += 1
        yield sample


# Per-process state for the worker pool. ``imap`` preserves input order, so the
# sample list -- and therefore a seeded training run -- is identical to the
# serial path.
_WORKER_STATE: dict[str, Any] = {}


def _worker_init(processor: Any, max_seq_len: int) -> None:
    _WORKER_STATE["processor"] = processor
    _WORKER_STATE["max_seq_len"] = max_seq_len
    # Each worker would otherwise start its own full-size thread pool and the
    # processes would fight over the same cores.
    torch.set_num_threads(1)


def _worker_encode(row: dict[str, Any]) -> tuple[Sample | None, str | None]:
    try:
        image_path, question, answer = extract_qa(row)
        sample = encode_example(
            _WORKER_STATE["processor"],
            image_path,
            question,
            answer,
            max_seq_len=_WORKER_STATE["max_seq_len"],
        )
    except SkipRow as exc:
        return None, exc.reason
    if sample is None:
        return None, None
    return sample, None


def _unshare(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Copy a tensor out of shared memory, if that is where it lives.

    torch's tensor IPC moves CPU tensors through ``/dev/shm`` and hands the
    parent a view of the same pages, so the storage stays charged to tmpfs for
    as long as the parent holds the sample. At ~25 MB per ``pixel_values`` that
    is 124 GB over a 5,000-row run, and encoding then dies with ``No space left
    on device`` -- on a shared box another user may be holding most of
    ``/dev/shm`` before the run starts. Cloning into anonymous heap costs one
    memcpy and caps the tmpfs footprint at whatever is in flight.
    """
    if tensor is None or not tensor.is_shared():
        return tensor
    return tensor.clone()


def _encode_rows_parallel(
    processor: Any,
    rows: Iterable[dict[str, Any]],
    max_seq_len: int,
    limit: int | None,
    skip_log: dict[str, int] | None,
    workers: int,
) -> Iterator[Sample]:
    import multiprocessing as mp
    from collections import deque

    context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    pool = context.Pool(
        processes=workers, initializer=_worker_init, initargs=(processor, max_seq_len)
    )
    # How many rows may be handed to the workers ahead of the caller. Every
    # result the workers have finished but the caller has not yet collected is
    # still charged to ``/dev/shm``, so *this window* -- not the size of the job --
    # is the tmpfs high-water mark. Left unbounded (``pool.imap``) it grows with
    # the job: measured 0.4 GB over 256 rows, 0.9 GB over 2,048, 48 GB over
    # 5,000, and the last of those dies on a box where another user is holding
    # most of ``/dev/shm``.
    #
    # A fresh pool per chunk would bound it too, but forking and re-pickling the
    # processor every chunk costs more than the encode saves (60.7 ms/row at 256
    # rows per pool, against ~11 ms/row here).
    window = max(2 * workers, 4)
    pending: deque = deque()
    remaining = iter(rows)
    yielded = 0

    def fill() -> None:
        """Keep the window full, without handing over more than ``limit`` rows."""
        while len(pending) < window:
            if limit is not None and yielded + len(pending) >= limit:
                return
            row = next(remaining, None)
            if row is None:
                return
            pending.append(pool.apply_async(_worker_encode, (row,)))

    try:
        fill()
        while pending:
            sample, reason = pending.popleft().get()
            fill()
            if sample is None:
                _record_skip(skip_log, reason)
                continue
            # ``fill`` sizes the window from ``yielded`` at submit time, and a row
            # it already handed over cannot be recalled, so the limit has to be
            # enforced here, where a sample is actually about to be produced.
            if limit is not None and yielded >= limit:
                return
            sample.pixel_values = _unshare(sample.pixel_values)
            yielded += 1
            yield sample
    finally:
        # Drop any uncollected results first, so their tmpfs storage is released
        # even on the abandoned-generator path.
        pending.clear()
        pool.terminate()
        pool.join()
