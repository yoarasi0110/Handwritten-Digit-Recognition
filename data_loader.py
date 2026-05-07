"""Dataset loading utilities for digit classification project."""

from typing import Tuple

import numpy as np
from sklearn.datasets import load_digits

ArrayPair = Tuple[np.ndarray, np.ndarray]


def load_sklearn_digits(normalize: bool = True) -> ArrayPair:
    """Load sklearn's 8x8 digits dataset.

    Args:
        normalize: Whether to scale pixel values to [0, 1].

    Returns:
        Tuple of features with shape (N, 8, 8) and labels with shape (N,).
    """
    dataset = load_digits()
    x = dataset.images.astype("float32")
    y = dataset.target.astype("int64")

    if normalize:
        x = x / 16.0

    return x, y


def load_mnist(normalize: bool = True) -> ArrayPair:
    """Load MNIST dataset via TensorFlow's built-in loader.

    Args:
        normalize: Whether to scale pixel values to [0, 1].

    Returns:
        Tuple of features with shape (N, 28, 28) and labels with shape (N,).
    """
    try:
        from tensorflow.keras.datasets import mnist
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TensorFlow is required to load the built-in MNIST dataset."
        ) from exc

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate([x_train, x_test], axis=0).astype("float32")
    y = np.concatenate([y_train, y_test], axis=0).astype("int64")

    if normalize:
        x = x / 255.0

    return x, y