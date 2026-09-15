"""Layer-0 lookahead scorer for HAWK visual-token pruning on Qwen2.5-VL.

Trains a small module (learned lookahead embeddings + additive LoRA on
layer-0 projections) that, appended after the prompt, reads the prompt once
through decoder layer 0 and then scores visual tokens with HAWK's own
aggregation. The learned scores replace HAWK's instruction-token query source
at pruning time while ``src/hawk`` stays byte-for-byte unchanged.
"""

from .config import LookhawkConfig, TEACHER_SELECTION_VALUES
from .modules import Layer0Scorer, LoraLinearAdd

__all__ = [
    "LookhawkConfig",
    "TEACHER_SELECTION_VALUES",
    "Layer0Scorer",
    "LoraLinearAdd",
]

__version__ = "0.1.0"
