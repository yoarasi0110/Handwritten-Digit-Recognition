"""數字分類的資料載入工具"""

from typing import Tuple

import numpy as np
from sklearn.datasets import load_digits

ArrayPair = Tuple[np.ndarray, np.ndarray]


def load_sklearn_digits(normalize: bool = True) -> ArrayPair:
    """載入 sklearn 內建 8x8 digits 資料集。

    參數：
        normalize：是否將像素值縮放到 [0, 1]。

    回傳：
        特徵 (N, 8, 8) 與標籤 (N,) 的 tuple。
    """
    dataset = load_digits()
    x = dataset.images.astype("float32") #x代表圖片(題目)
    y = dataset.target.astype("int64") #y代表標籤(答案)

    if normalize:
        x = x / 16.0

    return x, y


def load_mnist(normalize: bool = True) -> ArrayPair:
    """透過 TensorFlow 內建載入器讀取 MNIST 資料集。

    參數：
        normalize：是否將像素值縮放到 [0, 1]。

    回傳：
        特徵 (N, 28, 28) 與標籤 (N,) 的 tuple。
    """
    try:
        from tensorflow.keras.datasets import mnist
    except Exception as exc: 
        raise ImportError(
            "若要載入內建的 MNIST 資料集，必須安裝 TensorFlow"
        ) from exc

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate([x_train, x_test], axis=0).astype("float32")
    y = np.concatenate([y_train, y_test], axis=0).astype("int64")

    if normalize:
        x = x / 255.0

    return x, y
