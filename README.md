# Handwritten Digit Recognition Project

比較傳統機器學習（KNN）與神經網路（CNN）的手寫數字辨識專題架構。
*需使用python 3.12版本操作

## 專案結構

- `main.py`: 主流程，串接載入、前處理、訓練、評估與結果輸出
- `data_loader.py`: 載入 sklearn digits / TensorFlow 內建 MNIST 資料集
- `preprocess.py`: 資料形狀轉換（ML flatten / CNN reshape）
- `train_knn.py`: 訓練 KNN
- `train_cnn.py`: 訓練 CNN（需 TensorFlow）
- `evaluate.py`: Accuracy / report / confusion matrix / 訓練曲線
- `predict.py`: 單張圖片預測 helper
- `utils.py`: 建立資料夾、寫入報告
- `models/`: 儲存模型
- `results/`: 儲存結果

## 快速開始

```bash
pip install numpy scikit-learn matplotlib joblib tensorflow
python main.py --skip-cnn
```

上面會一次跑兩個資料集：
- knn-sklearn digits
- knn-mnist

若要同時跑四種模型（knn-sklearn, knn-mnist, cnn-sklearn, cnn-mnist）：

```bash
pip install tensorflow
python main.py
```

## 單一資料集模式

```bash
python main.py --single-dataset --dataset digits
python main.py --single-dataset --dataset mnist
```

## 輸出

- `results/accuracy.txt`
- `results/confusion_matrix_knn_digits.png`
- `results/confusion_matrix_knn_mnist.png`
- `results/confusion_matrix_cnn_digits.png`（有跑 CNN 才會有）
- `results/confusion_matrix_cnn_mnist.png`（有跑 CNN 才會有）
- `results/training_curve_digits.png`（有跑 CNN 才會有）
- `results/training_curve_mnist.png`（有跑 CNN 才會有）
- `models/knn_model_digits.pkl`
- `models/knn_model_mnist.pkl`
- `models/cnn_model_digits.h5`（有跑 CNN 才會有）
- `models/cnn_model_mnist.h5`（有跑 CNN 才會有）