"""Main pipeline for handwritten digit recognition project.

Compares KNN (traditional ML) and CNN (deep learning) on sklearn digits dataset.
"""

from __future__ import annotations

import argparse
import joblib
from sklearn.model_selection import train_test_split

from data_loader import load_sklearn_digits
from evaluate import evaluate_classifier, save_confusion_matrix, save_training_curve
from preprocess import flatten_for_ml, reshape_for_cnn
from train_cnn import train_cnn
from train_knn import train_knn
from utils import ensure_dirs, write_accuracy_report


def run_pipeline(skip_cnn: bool = False) -> None:
    ensure_dirs()
    x, y = load_sklearn_digits(normalize=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # KNN pipeline
    x_train_flat = flatten_for_ml(x_train)
    x_test_flat = flatten_for_ml(x_test)
    knn_result = train_knn(x_train_flat, y_train, n_neighbors=3)
    knn_eval = evaluate_classifier(knn_result.model, x_test_flat, y_test)
    joblib.dump(knn_result.model, "models/knn_model.pkl")
    save_confusion_matrix(knn_eval.confusion, "results/confusion_matrix_knn.png")

    lines = [
        "=== Digit Classification Comparison ===",
        f"KNN accuracy: {knn_eval.accuracy:.4f}",
        f"KNN train time (s): {knn_result.train_seconds:.4f}",
        f"KNN infer time (s): {knn_eval.infer_seconds:.4f}",
        "",
        "KNN classification report:",
        knn_eval.report,
    ]

    if not skip_cnn:
        x_train_cnn = reshape_for_cnn(x_train)
        x_test_cnn = reshape_for_cnn(x_test)
        cnn_result = train_cnn(x_train_cnn, y_train, epochs=8)
        cnn_eval = evaluate_classifier(cnn_result.model, x_test_cnn, y_test)
        cnn_result.model.save("models/cnn_model.h5")
        save_confusion_matrix(cnn_eval.confusion, "results/confusion_matrix_cnn.png")
        save_training_curve(cnn_result.history, "results/training_curve.png")
        lines += [
            "",
            f"CNN accuracy: {cnn_eval.accuracy:.4f}",
            f"CNN train time (s): {cnn_result.train_seconds:.4f}",
            f"CNN infer time (s): {cnn_eval.infer_seconds:.4f}",
            "",
            "CNN classification report:",
            cnn_eval.report,
        ]

    report = "\n".join(lines)
    write_accuracy_report(report)
    print(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Digit recognition comparison project")
    parser.add_argument("--skip-cnn", action="store_true", help="Run only traditional ML (KNN) pipeline")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(skip_cnn=args.skip_cnn)
