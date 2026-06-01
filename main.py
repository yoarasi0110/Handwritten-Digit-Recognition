"""手寫數字辨識主程式

比較 KNN(傳統機器學習)與 CNN(深度學習)在 sklearn digits 與 MNIST 資料集上的表現
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

#決定要載哪個資料集
DATASET_LOADERS = {
    "digits": load_sklearn_digits,
    "mnist": load_mnist,
}

#對指定資料集執行訓練與評估流程，回傳結果文字列表，一次跑一個資料集
def run_for_dataset(dataset: str, skip_cnn: bool = False) -> list[str]:
    x, y = DATASET_LOADERS[dataset](normalize=True)
    #把資料分成訓練集和測試集，20%當測試集，並且保持類別分布一致
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    lines = [f"=== Dataset: {dataset} ==="]

    # KNN pipeline
    #把資料攤平ex: 8*8的圖變成64維的向量，然後用 KNN 訓練模型，接著評估模型表現，最後把模型存起來，還有把混淆矩陣畫出來存成圖片
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
    # CNN pipeline
    if not skip_cnn:
        #CNN 要保留圖片結構，所以不會攤平，而是整理成像(批次數,高,寬,通道數)ex.(100, 8, 8, 1)
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

#整體流程控制，決定要跑哪個資料集，要不要跳過 CNN，最後把結果寫成報告
def run_pipeline(skip_cnn: bool = False, dataset: str = "digits", run_all: bool = True) -> None:
    ensure_dirs() #確保模型和結果的資料夾存在
    datasets = ["digits", "mnist"] if run_all else [dataset]

    report_lines = ["=== Digit Classification Comparison ==="]
    for ds in datasets:
        report_lines += [""] + run_for_dataset(ds, skip_cnn=skip_cnn)

    report = "\n".join(report_lines)
    write_accuracy_report(report)
    print(report)

#讀命令列參數，決定要跳過 CNN、要跑哪個資料集、要不要只跑一個資料集
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Digit recognition comparison project")
    parser.add_argument("--skip-cnn", action="store_true", help="Run only traditional ML (KNN) pipeline")
    parser.add_argument("--dataset", choices=["digits", "mnist"], default="digits", help="Choose dataset in single mode")
    parser.add_argument("--single-dataset", action="store_true", help="Run only one dataset instead of both datasets")
    return parser.parse_args()

#主程式入口，解析命令列參數，然後執行整體流程
if __name__ == "__main__":
    args = parse_args()
    run_pipeline(skip_cnn=args.skip_cnn, dataset=args.dataset, run_all=not args.single_dataset)
