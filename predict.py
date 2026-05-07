"""Single-image prediction helpers."""

from __future__ import annotations

import numpy as np


def predict_single_ml(model, image_2d: np.ndarray) -> int:
    x = image_2d.reshape(1, -1)
    return int(model.predict(x)[0])


def predict_single_cnn(model, image_2d: np.ndarray) -> int:
    x = image_2d.reshape(1, image_2d.shape[0], image_2d.shape[1], 1).astype("float32")
    pred = model.predict(x, verbose=0)
    return int(np.argmax(pred, axis=1)[0])
