"""Evaluation utilities for classification models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass
class EvalResult:
    accuracy: float
    report: str
    confusion: np.ndarray
    infer_seconds: float


def evaluate_classifier(model: Any, x_test: np.ndarray, y_test: np.ndarray) -> EvalResult:
    start = perf_counter()
    y_pred = model.predict(x_test)
    infer = perf_counter() - start
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return EvalResult(acc, report, cm, infer)


def save_confusion_matrix(cm: np.ndarray, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def save_training_curve(history: Any, output_path: str) -> None:
    hist = history.history
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(hist.get("accuracy", []), label="train_acc")
    plt.plot(hist.get("val_accuracy", []), label="val_acc")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
