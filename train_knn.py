"""KNN的訓練模組"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class KNNTrainResult:
    model: KNeighborsClassifier
    train_seconds: float

#用訓練資料訓練 KNN，最後把模型和訓練時間回傳
def train_knn(x_train: np.ndarray, y_train: np.ndarray, n_neighbors: int = 3) -> KNNTrainResult:
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    start = perf_counter()
    #訓練 KNN 模型，KNN 不像神經網路那樣更新大量權重，它比較像是把訓練資料存起來，建立之後分類要用的結構
    model.fit(x_train, y_train)
    elapsed = perf_counter() - start
    return KNNTrainResult(model=model, train_seconds=elapsed)
