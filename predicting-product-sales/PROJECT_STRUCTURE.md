# PROJECT_STRUCTURE.md — Predicting Product Sales

## Mục tiêu & ràng buộc
Dự đoán doanh số (hồi quy). Python 3.10+, chỉ numpy / pandas / matplotlib / pytest.
Mô hình tự cài đặt; KHÔNG scikit-learn. Không rò rỉ dữ liệu: mọi thống kê
(median, IQR, mã hóa, đếm theo khách/sản phẩm) chỉ học (fit) trên tập train.

## Cấu trúc thư mục
product-sales-prediction/
├── PROJECT_STRUCTURE.md        # file đặc tả này
├── README.md                   # cách cài, chạy, kết quả
├── requirements.txt            # numpy, pandas, matplotlib, pytest (cấm scikit-learn)
├── .gitignore
├── data/
│   ├── raw/                    # CSV gốc tải từ Kaggle (không sửa tay)
│   └── processed/              # train.csv, test.csv sau xử lý (cột cuối = target)
├── save_models/                # preprocessor.npy, <model>.npy, best_model.txt, best_params.json
├── reports/
│   ├── figures/                # hình EDA + hình đánh giá
│   ├── results.csv             # bảng MSE/MAE/RMSE/R2 trên test
│   ├── tuning_results.csv      # kết quả grid search
│   └── REPORT.md               # báo cáo
├── src/
│   ├── __init__.py
│   ├── config.py               # đường dẫn, seed, tên cột, tham số chung
│   ├── metrics.py              # mse, mae, rmse, r2, evaluate_all
│   ├── tuning.py               # k_fold_indices, cross_val_score, grid_search
│   ├── predict.py              # SalesPredictor: nạp preprocessor + model, dự đoán
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # load_raw: đọc CSV, kiểm tra cột, parse ngày
│   │   ├── cleaning.py         # drop_invalid_rows + Cleaner (thiếu, ngoại lai)
│   │   ├── split.py            # train_test_split (theo thời gian hoặc ngẫu nhiên)
│   │   └── encoding.py         # CategoricalEncoder (one-hot, fit trên train)
│   ├── features/
│   │   ├── __init__.py
│   │   ├── seasonal.py         # tháng/quý/thứ/cuối tuần/lễ + sin-cos
│   │   ├── popularity.py       # PopularityFeatures: reviews, rating, tần suất sản phẩm
│   │   └── customer_value.py   # CustomerValueFeatures: CLV, tần suất mua
│   └── models/
│       ├── __init__.py
│       ├── base.py             # BaseRegressor (fit/predict/get_params/set_params/clone) + save_model/load_model (.npy)
│       ├── linear_regression.py    # tự cài (ridge, nghiệm đóng)
│       ├── decision_tree.py        # tự cài (CART hồi quy)
│       ├── random_forest.py        # tự cài (bagging + chọn đặc trưng ngẫu nhiên)
│       └── gradient_boosting.py    # tự cài (boosting trên residual, loss MSE)
├── scripts/
│   ├── explore_data.py        # EDA: thống kê + hình
│   ├── prepare_data.py        # load → clean → split → features → encode → lưu
│   ├── train.py               # tune (k-fold) + train 4 mô hình, lưu .npy
│   ├── evaluate.py            # đánh giá trên test, xuất bảng + hình
│   └── predict_scenario.py    # CLI dự đoán theo kịch bản
└── tests/
    ├── test_metrics.py
    └── test_models.py

## Luồng dữ liệu
data/raw → loader → drop_invalid_rows → split → Cleaner(fit train) →
seasonal + PopularityFeatures + CustomerValueFeatures (fit train) →
CategoricalEncoder (fit train) → data/processed/{train,test}.csv + save_models/preprocessor.npy
→ train.py (k-fold tuning) → save_models/<model>.npy → evaluate.py (test) → reports/
Dự đoán: kịch bản (dict/JSON) → SalesPredictor (preprocessor.npy + best model) → doanh số

## Giao diện chính (các module phải khớp)
- loader.load_raw(path: Path | None = None) -> pd.DataFrame
- cleaning.drop_invalid_rows(df) -> pd.DataFrame
- cleaning.Cleaner: fit(df) -> self; transform(df) -> pd.DataFrame
- split.train_test_split(df, test_size, seed, date_col=None) -> (train_df, test_df)
- seasonal.add_seasonal_features(df, date_col, holiday_col=None) -> pd.DataFrame
- PopularityFeatures / CustomerValueFeatures: fit(train_df) -> self; transform(df) -> pd.DataFrame
- encoding.CategoricalEncoder: fit(df, cols) -> self; transform(df) -> pd.DataFrame
- save_models/preprocessor.npy = dict {"cleaner","popularity","customer","encoder","feature_columns"}
- metrics: mse, mae, rmse, r2 (y_true, y_pred) -> float; evaluate_all -> dict
- BaseRegressor: fit(X, y) -> self; predict(X) -> np.ndarray; get_params(); set_params(**p); clone()
- Mô hình: LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor
- tuning.grid_search(model, param_grid, X, y, k, metric="rmse", seed=42) -> (best_params, results_df)
- predict.SalesPredictor.load(models_dir) ; .predict(df_raw) -> np.ndarray

## Thứ tự dựng
1. Khung dự án — mọi prompt sau cần đường dẫn có sẵn
2. config.py → 3. loader.py → 4. explore_data.py — hiểu dữ liệu trước khi xử lý
5. cleaning.py → 6. split.py → 7. encoding.py — dữ liệu sạch, chia, mã hóa
8–10. seasonal / popularity / customer_value — đặc trưng
11. prepare_data.py — nối toàn bộ tiền xử lý
12. metrics.py → 13. base.py → 14–17. 4 mô hình → 18. tuning.py
19. train.py → 20. evaluate.py → 21. predict.py → 22. predict_scenario.py
23–24. tests → 25. REPORT.md → 26. README.md

## Lưu ý / giả định cần xác nhận
- Có được dùng scikit-learn không? (mặc định: KHÔNG)
