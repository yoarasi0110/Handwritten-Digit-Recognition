"""KNN 與 CNN 模型的前處理工具"""

from __future__ import annotations

from typing import Tuple

import numpy as np

#表示資料集分割後的四個部分：訓練特徵、訓練標籤、測試特徵、測試標籤
SplitSet = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def flatten_for_ml(x: np.ndarray) -> np.ndarray:
    #將影像張量由 (N, H, W)(有幾張圖, 高, 寬)轉為 (N, H*W)(有幾張圖, 每張圖的像素數)
    return x.reshape(x.shape[0], -1)


def reshape_for_cnn(x: np.ndarray) -> np.ndarray:
    #將影像張量由 (N, H, W) 轉為 (N, H, W, 1)(有幾張圖, 高, 寬, 通道數)，其中通道數為 1，表示灰階圖像
    return x.reshape((-1, x.shape[1], x.shape[2], 1)).astype("float32")
