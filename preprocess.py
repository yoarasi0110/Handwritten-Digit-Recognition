"""Preprocessing helpers for traditional ML and CNN models."""

from __future__ import annotations

from typing import Tuple

import numpy as np


SplitSet = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def flatten_for_ml(x: np.ndarray) -> np.ndarray:
    """Convert image tensors from (N, H, W) to (N, H*W)."""
    return x.reshape(x.shape[0], -1)


def reshape_for_cnn(x: np.ndarray) -> np.ndarray:
    """Convert image tensors from (N, H, W) to (N, H, W, 1)."""
    return x.reshape((-1, x.shape[1], x.shape[2], 1)).astype("float32")
