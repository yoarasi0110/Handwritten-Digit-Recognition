"""Training module for CNN model.

TensorFlow is optional; this module raises a clear error if it's missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception:  # pragma: no cover - environment-dependent import
    tf = None
    layers = None
    models = None


@dataclass
class CNNTrainResult:
    model: Any
    history: Any
    train_seconds: float


def _build_cnn(input_shape: tuple[int, int, int]):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 5,
    batch_size: int = 32,
    validation_split: float = 0.1,
) -> CNNTrainResult:
    """Train a small CNN classifier."""
    if tf is None:
        raise ImportError("TensorFlow is not installed. Install tensorflow to train CNN.")

    model = _build_cnn((x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    start = perf_counter()
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0,
    )
    elapsed = perf_counter() - start
    return CNNTrainResult(model=model, history=history, train_seconds=elapsed)
