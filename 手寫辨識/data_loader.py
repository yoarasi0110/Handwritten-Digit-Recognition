"""Dataset loading utilities for digit classification project."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.datasets import load_digits


ArrayPair = Tuple[np.ndarray, np.ndarray]


def load_sklearn_digits(normalize: bool = True) -> ArrayPair:
    """Load sklearn's 8x8 digits dataset.

    Args:
        normalize: Whether to scale pixel values to [0, 1].

    Returns:
        Tuple of features (N, 8, 8) and labels (N,).
    """
    dataset = load_digits()
    x = dataset.images.astype("float32")
    if normalize:
        x = x / 16.0
    y = dataset.target.astype("int64")
    return x, y
