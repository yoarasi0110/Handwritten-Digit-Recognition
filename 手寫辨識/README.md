# Handwritten Digit Recognition Project

比較傳統機器學習（KNN）與神經網路（CNN）的手寫數字辨識專題架構。

## 專案結構

- `main.py`: 主流程，串接載入、前處理、訓練、評估與結果輸出
- `data_loader.py`: 載入 sklearn digits 資料集
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
pip install numpy scikit-learn matplotlib joblib
python main.py --skip-cnn
```

若要跑 CNN：

```bash
pip install tensorflow
python main.py
```

## 輸出

- `results/accuracy.txt`
- `results/confusion_matrix_knn.png`
- `results/confusion_matrix_cnn.png`（有跑 CNN 才會有）
- `results/training_curve.png`（有跑 CNN 才會有）
