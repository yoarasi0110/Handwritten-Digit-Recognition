"""CNN 模型訓練模組"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

#嘗試導入 TensorFlow 和 Keras，如果失敗則將相關變量設置為 None，以便在使用時引發 ImportError
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception:
    tf = None
    layers = None
    models = None


@dataclass
class CNNTrainResult:
    model: Any
    history: Any
    train_seconds: float

#建立一個 CNN 模型架構，包含兩個卷積層和兩個池化層，最後是全連接層和輸出層
def _build_cnn(input_shape: tuple[int, int, int]):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            #從圖片中抓局部特徵，例如邊緣、線條、簡單筆畫
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            #把特徵圖縮小，減少計算量，也保留重要特徵
            layers.MaxPooling2D((2, 2)),
            #學更複雜的特徵
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            #再縮小特徵圖
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),#把特徵圖攤平成一維向量，準備輸入全連接層
            layers.Dense(64, activation="relu"),#全連接層，學習特徵之間的關係
            layers.Dense(10, activation="softmax"),#輸出層，10 類數字的機率分布
        ]
    )
    #用 Adam 來更新權重，用 sparse_categorical_crossentropy 作為損失函數，評估指標是準確率
    #Adam是一種用來更新神經網路權重的最佳化方法
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

#用訓練資料去真正訓練 CNN 模型，並測量訓練時間
def train_cnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 5,
    batch_size: int = 32,
    #從訓練資料中拿出 10% 當驗證集，評估模型在訓練過程中的表現
    #訓練過程中拿來觀察模型有沒有進步、會不會過擬合
    validation_split: float = 0.1,
) -> CNNTrainResult:
    
    if tf is None:
        raise ImportError("TensorFlow is not installed. Install tensorflow to train CNN.")

    model = _build_cnn((x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    start = perf_counter()
    #訓練模型，使用訓練資料，設定訓練的輪數、批次大小和驗證集比例，並且不顯示訓練過程的詳細信息
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
