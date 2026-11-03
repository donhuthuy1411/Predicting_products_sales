# Predicting Product Sales — Practice 2

Dự đoán doanh số sản phẩm (hồi quy) bằng Python thuần: tiền xử lý dữ liệu, tạo đặc trưng, tự cài đặt 4 mô hình bằng **numpy** (không dùng scikit-learn), tinh chỉnh bằng grid search + k-fold tự viết, đánh giá MSE/MAE/RMSE/R2 và "triển khai" bằng `SalesPredictor` + CLI kịch bản.

## Yêu cầu

- Python 3.10+
- Cài thư viện:

```bash
pip install -r requirements.txt   # numpy, pandas, matplotlib, pytest — KHÔNG dùng scikit-learn
```

## Dữ liệu

Tải dataset **BigMart Sales** từ Kaggle (file gốc tên `Train.csv`, 8523 dòng) và **đổi tên thành `bigmart_sales.csv`** đặt vào `data/raw/bigmart_sales.csv`:

- Kaggle: https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-dataset (chọn file `Train.csv` rồi đổi tên)
- Hoặc bản mirror đã dùng: https://raw.githubusercontent.com/pydeveloperashish/BigMart-Sales-Prediction-With-Deployment/main/dataset/Train.csv (đổi tên khi lưu thành `bigmart_sales.csv`)

Nếu dùng dataset khác: khai báo lại tên cột trong `src/config.py` (cột nào không có thì để `None`).

## Thứ tự chạy (từ thư mục gốc)

```bash
python -m scripts.explore_data        # EDA: thống kê + hình vào reports/figures/
python -m scripts.prepare_data        # tiền xử lý -> data/processed/{train,test}.csv + save_models/preprocessor.npy
python -m scripts.train --fast        # chạy nhanh: tune trên 20% train (kiểm tra thời gian)
python -m scripts.train               # bản đầy đủ: tune 5-fold trên toàn bộ train
python -m scripts.evaluate            # đánh giá trên test -> reports/results.csv + hình
python -m scripts.predict_scenario --demo   # dự đoán theo kịch bản mẫu
python -m pytest -q                   # chạy test
```

## Kết quả (tập test, RMSE nhỏ nhất chọn mô hình tốt nhất)

| model | mse | mae | rmse | r2 |
|---|---|---|---|---|
| **gradient_boosting** | 1115280.05 | 740.56 | **1056.07** | **0.6118** |
| random_forest | 1135939.87 | 745.86 | 1065.80 | 0.6046 |
| decision_tree | 1152400.45 | 752.64 | 1073.50 | 0.5989 |
| linear_regression | 1236646.47 | 823.42 | 1112.05 | 0.5696 |

Chi tiết xem `reports/REPORT.md`, bảng đầy đủ ở `reports/results.csv` và `reports/tuning_results.csv`.

## Cấu trúc thư mục

```
├── PROJECT_STRUCTURE.md   # đặc tả cấu trúc + giao diện các module
├── README.md
├── requirements.txt       # numpy, pandas, matplotlib, pytest (cấm scikit-learn)
├── data/raw/              # CSV gốc từ Kaggle (không sửa tay)
├── data/processed/        # train.csv, test.csv sau xử lý (cột cuối = target)
├── save_models/           # preprocessor.npy, <model>.npy, best_model.txt, best_params.json
├── reports/               # REPORT.md, results.csv, tuning_results.csv, figures/
├── src/                   # config, metrics, tuning, predict + data/ features/ models/
├── scripts/               # explore_data, prepare_data, train, evaluate, predict_scenario
└── tests/                 # test_metrics.py, test_models.py
```

Lưu ý: các mô hình (Linear Regression, Decision Tree, Random Forest, Gradient Boosting), k-fold CV và grid search đều **tự cài đặt bằng numpy**, không dùng scikit-learn.
