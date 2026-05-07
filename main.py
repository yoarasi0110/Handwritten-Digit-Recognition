"""Main pipeline for handwritten digit recognition project.

Compares KNN (traditional ML) and CNN (deep learning) on sklearn digits and MNIST datasets.
"""

from __future__ import annotations

import argparse
import joblib
from sklearn.model_selection import train_test_split

from data_loader import load_mnist, load_sklearn_digits
from evaluate import evaluate_classifier, save_confusion_matrix, save_training_curve
from preprocess import flatten_for_ml, reshape_for_cnn
from train_cnn import train_cnn
from train_knn import train_knn
from utils import ensure_dirs, write_accuracy_report


DATASET_LOADERS = {
    "digits": load_sklearn_digits,
    "mnist": load_mnist,
}


def run_for_dataset(dataset: str, skip_cnn: bool = False) -> list[str]:
    x, y = DATASET_LOADERS[dataset](normalize=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    lines = [f"=== Dataset: {dataset} ==="]

    # KNN pipeline
    x_train_flat = flatten_for_ml(x_train)
    x_test_flat = flatten_for_ml(x_test)
    knn_result = train_knn(x_train_flat, y_train, n_neighbors=3)
    knn_eval = evaluate_classifier(knn_result.model, x_test_flat, y_test)
    joblib.dump(knn_result.model, f"models/knn_model_{dataset}.pkl")
    save_confusion_matrix(knn_eval.confusion, f"results/confusion_matrix_knn_{dataset}.png")

    lines += [
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
        cnn_result.model.save(f"models/cnn_model_{dataset}.keras")
        save_confusion_matrix(cnn_eval.confusion, f"results/confusion_matrix_cnn_{dataset}.png")
        save_training_curve(cnn_result.history, f"results/training_curve_{dataset}.png")
        lines += [
            "",
            f"CNN accuracy: {cnn_eval.accuracy:.4f}",
            f"CNN train time (s): {cnn_result.train_seconds:.4f}",
            f"CNN infer time (s): {cnn_eval.infer_seconds:.4f}",
            "",
            "CNN classification report:",
            cnn_eval.report,
        ]

    return lines


def run_pipeline(skip_cnn: bool = False, dataset: str = "digits", run_all: bool = True) -> None:
    ensure_dirs()
    datasets = ["digits", "mnist"] if run_all else [dataset]

    report_lines = ["=== Digit Classification Comparison ==="]
    for ds in datasets:
        report_lines += [""] + run_for_dataset(ds, skip_cnn=skip_cnn)

    report = "\n".join(report_lines)
    write_accuracy_report(report)
    print(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Digit recognition comparison project")
    parser.add_argument("--skip-cnn", action="store_true", help="Run only traditional ML (KNN) pipeline")
    parser.add_argument("--dataset", choices=["digits", "mnist"], default="digits", help="Choose dataset in single mode")
    parser.add_argument("--single-dataset", action="store_true", help="Run only one dataset instead of both datasets")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(skip_cnn=args.skip_cnn, dataset=args.dataset, run_all=not args.single_dataset)