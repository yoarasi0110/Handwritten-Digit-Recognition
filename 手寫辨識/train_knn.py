"""Training module for traditional baseline models."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class KNNTrainResult:
    model: KNeighborsClassifier
    train_seconds: float


def train_knn(x_train: np.ndarray, y_train: np.ndarray, n_neighbors: int = 3) -> KNNTrainResult:
    """Train a KNN classifier and return model plus training time."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    start = perf_counter()
    model.fit(x_train, y_train)
    elapsed = perf_counter() - start
    return KNNTrainResult(model=model, train_seconds=elapsed)
