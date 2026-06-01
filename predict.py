"""單張影像預測工具"""

from __future__ import annotations

import numpy as np

#載入訓練好的KNN模型，對單張圖片進行預測
def predict_single_ml(model, image_2d: np.ndarray) -> int:
    x = image_2d.reshape(1, -1)
    return int(model.predict(x)[0])

#載入訓練好的 CNN 模型，對單張圖片進行預測
def predict_single_cnn(model, image_2d: np.ndarray) -> int:
    x = image_2d.reshape(1, image_2d.shape[0], image_2d.shape[1], 1).astype("float32")
    pred = model.predict(x, verbose=0)
    #預測結果是機率分布，取最大值的索引就是預測的類別，ex.如果 pred 是 [[0.1, 0.7, 0.2]]，那麼 np.argmax(pred) 就會回傳 1，表示預測為類別 1（對應數字 1）
    return int(np.argmax(pred, axis=1)[0])
