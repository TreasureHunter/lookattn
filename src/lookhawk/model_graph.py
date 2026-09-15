"""Version-agnostic access to Qwen2.5-VL submodules used by the scorer.

Transformers changed the Qwen2.5-VL nesting between releases:

* 4.51.x (flat): outer ``Qwen2_5_VLForConditionalGeneration`` holds
  ``.visual`` and ``.model`` = text decoder (``embed_tokens``/``layers``).
* 4.52.x+ (wrapper, what HAWK and this repo pin): ``outer.model`` is a
  ``Qwen2_5_VLModel`` wrapper ``W`` holding ``.visual``, ``.language_model``
  (text decoder with ``embed_tokens``/``layers``) and the ``rope_deltas``
  cache. HAWK stores ``_hawk_*`` state on ``W`` and the patched prefill calls
  ``prune_qwen_prefill(model=W, ...)``.

The scorer only ever touches decoder layer 0's projections/MLP plus (for
training) the visual tower and the token embedding table, so we resolve those
handles once and never hard-code an attribute chain elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_END_TOKEN_ID = 151653
IM_END_TOKEN_ID = 151645

DEFAULT_HEADS = 28
DEFAULT_KV_HEADS = 4
DEFAULT_HEAD_DIM = 128
DEFAULT_HIDDEN = 3584


@dataclass(frozen=True)
class ModelHandle:
    """Resolved submodules shared by training and inference."""

    wrapper: Any  # state-owner ``W`` (the object ``prune_qwen_prefill`` receives).
    text: Any  # text decoder with ``embed_tokens`` and ``layers``.
    visual: Any  # vision tower or ``None`` when only the text path is needed.

    def layer0(self) -> Any:
        return self.text.layers[0]

    def embed_tokens(self) -> Any:
        return self.text.embed_tokens

    def config(self) -> Any:
        return getattr(self.wrapper, "config", None) or getattr(self.text, "config", None)


def _has(obj: Any, name: str) -> bool:
    return obj is not None and hasattr(obj, name)


def _find_text(obj: Any) -> Any | None:
    """Locate an object exposing ``embed_tokens`` and ``layers``."""
    if obj is None:
        return None
    for attr in ("language_model", "model"):
        candidate = getattr(obj, attr, None)
        if _has(candidate, "embed_tokens") and _has(candidate, "layers"):
            return candidate
    if _has(obj, "embed_tokens") and _has(obj, "layers"):
        return obj
    return None


def _find_visual(obj: Any) -> Any | None:
    """Locate the vision tower, scanning a few known homes."""
    if obj is None:
        return None
    for attr in ("visual",):
        candidate = getattr(obj, attr, None)
        if candidate is not None and not _has(candidate, "embed_tokens"):
            return candidate
    # 4.51 style: visual sits on the outer model, not on the text decoder.
    inner = getattr(obj, "model", None)
    if inner is not None and _has(inner, "visual"):
        return inner.visual
    return None


def _is_text_decoder(obj: Any) -> bool:
    return _has(obj, "embed_tokens") and _has(obj, "layers")


def resolve_handle(model: Any) -> ModelHandle:
    """Resolve wrapper/text/visual from an outer, wrapper, or bare model.

    Handles the known topologies without hard-coding a single chain:

    * 4.52 outer ``Qwen2_5_VLForConditionalGeneration`` whose ``.model`` is the
      ``Qwen2_5_VLModel`` wrapper ``W`` holding ``.visual`` and
      ``.language_model`` (the text decoder). Pruning state lives on ``W``.
    * 4.52 ``W`` passed directly (what ``prune_qwen_prefill`` receives).
    * 4.51 flat layout where ``.model`` *is* the text decoder.

    ``wrapper`` is always the state-owner object HAWK stores ``_hawk_*`` on --
    for the 4.52 outer model that is ``model.model``, never the outer object
    itself (see the Case B comment below).
    """
    if model is None:
        raise TypeError("resolve_handle requires a model object")

    def resolve(owner: Any) -> ModelHandle:
        text = getattr(owner, "language_model", None)
        if not _is_text_decoder(text):
            text = getattr(owner, "model", None)
        if not _is_text_decoder(text):
            text = owner
        if not _is_text_decoder(text):
            raise TypeError(
                "could not locate a Qwen2.5-VL text decoder (expected "
                "`embed_tokens`/`layers` under `.language_model`, `.model`, or "
                "the object itself)"
            )
        visual = _find_visual(owner)
        if visual is None:
            visual = _find_visual(text)
        return ModelHandle(wrapper=owner, text=text, visual=visual)

    # Case B first: the outer conditional-generation model. It cannot be told
    # apart from ``W`` by ``hasattr(model, "language_model")`` -- it *defines*
    # that attribute as a forwarding property (4.52
    # ``Qwen2_5_VLForConditionalGeneration.language_model -> self.model.language_model``),
    # so ``hasattr`` is true for both and the outer object would be mistaken for
    # the state owner. Install state written to it is invisible to
    # ``prune_qwen_prefill``, which is handed ``W``; and ``get_rope_index`` only
    # exists on ``W``. Detect the shape instead: an outer model is an object
    # whose ``.model`` is itself a wrapper over a text decoder.
    inner = getattr(model, "model", None)
    if _is_text_decoder(getattr(inner, "language_model", None)):
        return resolve(inner)
    # Case A: the object itself is the state-owner wrapper ``W`` (has
    # ``language_model``/``.visual``), or is already a text decoder, or is the
    # 4.51 flat outer whose ``.model`` *is* the text decoder.
    return resolve(model)


def resolve_layer0(model: Any) -> Any:
    """Direct helper used by inference when handed the prune-time object."""
    return resolve_handle(model).layer0()


def verify_model_graph(handle: ModelHandle) -> dict[str, Any]:
    """Assert the Qwen2.5-VL-7B geometry the scorer relies on.

    Raises a descriptive ``RuntimeError`` (rather than failing later mid-step)
    when a wrong transformers version or an unexpected config is loaded.
    """
    import torch

    text = handle.text
    config = handle.config()
    layer0 = text.layers[0]
    attention = layer0.self_attn
    num_layers = len(text.layers)
    info = {
        "num_layers": num_layers,
        "num_heads": int(attention.num_heads),
        "num_kv_heads": int(attention.num_key_value_heads),
        "num_kv_groups": int(attention.num_key_value_groups),
        "head_dim": int(attention.head_dim),
        "hidden_size": int(attention.hidden_size)
        if _has(attention, "hidden_size")
        else int(attention.q_proj.in_features),
        "q_out": int(attention.q_proj.out_features),
        "k_out": int(attention.k_proj.out_features),
        "image_token_id": int(getattr(config, "image_token_id", IMAGE_TOKEN_ID)),
        "cuda": (
            f"{torch.cuda.get_device_name(0)} "
            f"cap={torch.cuda.get_device_capability(0)}"
            if torch.cuda.is_available()
            else "none"
        ),
    }
    if num_layers != 28:
        raise RuntimeError(f"expected 28 decoder layers, got {num_layers}")
    if int(attention.num_heads) != DEFAULT_HEADS:
        raise RuntimeError(
            f"expected {DEFAULT_HEADS} Q heads, got {attention.num_heads}"
        )
    if int(attention.num_key_value_heads) != DEFAULT_KV_HEADS:
        raise RuntimeError(
            f"expected {DEFAULT_KV_HEADS} KV heads, got {attention.num_key_value_heads}"
        )
    if int(attention.head_dim) != DEFAULT_HEAD_DIM:
        raise RuntimeError(
            f"expected head_dim={DEFAULT_HEAD_DIM}, got {attention.head_dim}"
        )
    if info["hidden_size"] != DEFAULT_HIDDEN:
        raise RuntimeError(
            f"expected hidden_size={DEFAULT_HIDDEN}, got {info['hidden_size']}"
        )
    if int(attention.num_key_value_groups) != DEFAULT_HEADS // DEFAULT_KV_HEADS:
        raise RuntimeError(
            f"expected {DEFAULT_HEADS // DEFAULT_KV_HEADS} KV groups, "
            f"got {attention.num_key_value_groups}"
        )
    return info
