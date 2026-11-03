# Đặc tả toàn bộ project — Predicting Product Sales

## 1. Mục tiêu & ràng buộc

- Bài toán: dự đoán **doanh số sản phẩm** (biến mục tiêu liên tục → **hồi quy**).
- Pipeline: chuẩn bị dữ liệu → tạo đặc trưng → huấn luyện/tinh chỉnh 4 mô hình (Linear Regression, Decision Tree, Random Forest, Gradient Boosting) → đánh giá bằng MSE/MAE/RMSE/R2 → "triển khai" bằng class `SalesPredictor` + CLI kịch bản (`scripts/predict_scenario.py`).
- Python 3.10+, chỉ dùng **numpy / pandas / matplotlib / pytest**. Mô hình **tự cài đặt bằng numpy — KHÔNG dùng scikit-learn** trong `src/` và `scripts/`.
- Không rò rỉ dữ liệu: mọi thống kê (median, IQR, mode, tần suất sản phẩm, bảng one-hot) chỉ được học (fit) trên tập train.

## 2. Dataset & cấu hình cột

- Dataset: **BigMart Sales** (Kaggle; file gốc `Train.csv` đổi tên thành `bigmart_sales.csv`), file `data/raw/bigmart_sales.csv` — 8523 dòng × 12 cột, không sửa tay. Tải từ Kaggle hoặc mirror GitHub (xem `README.md`).
- Mọi tên cột khai báo tập trung ở `src/config.py`:
  - `TARGET_COL = "Item_Outlet_Sales"`
  - `NUMERIC_COLS = [Item_Weight, Item_Visibility, Item_MRP, Outlet_Establishment_Year]`
  - `CATEGORICAL_COLS = [Item_Fat_Content, Item_Type, Outlet_Identifier, Outlet_Size, Outlet_Location_Type, Outlet_Type]`
  - `PRODUCT_ID_COL = "Item_Identifier"`, `PRICE_COL = "Item_MRP"`
  - Các cột không có trong bộ này đặt `None`: `DATE_COL, HOLIDAY_COL, CUSTOMER_ID_COL, QUANTITY_COL, REVIEW_COL, RATING_COL, AD_SPEND_COL` → các module tương ứng tự bỏ qua mà không lỗi.
- Tham số chung: `SEED = 42`, `TEST_SIZE = 0.2`, `K_FOLDS = 5`.
- Không có cột ngày → chia train/test **ngẫu nhiên** (nếu có cột ngày thì chia theo thời gian, test nằm sau train).

## 3. Cấu trúc thư mục

```
├── PROJECT_STRUCTURE.md     # đặc tả gốc (giữ nguyên văn)
├── README.md                # cách cài, chạy, kết quả
├── requirements.txt         # numpy, pandas, matplotlib, pytest (cấm scikit-learn)
├── docs/                    # tài liệu này + docs/Algorithm/*.md
├── data/
│   ├── raw/bigmart_sales.csv  # CSV gốc từ Kaggle (đổi tên từ Train.csv)
│   └── processed/           # train.csv, test.csv (cột cuối = target)
├── save_models/             # preprocessor.npy, <model>.npy, best_model.txt, best_params.json
├── reports/                 # REPORT.md, results.csv, tuning_results.csv, figures/
├── src/
│   ├── config.py            # đường dẫn, seed, tên cột, tham số chung
│   ├── metrics.py           # mse, mae, rmse, r2, evaluate_all
│   ├── tuning.py            # k_fold_indices, cross_val_score, grid_search
│   ├── predict.py           # SalesPredictor (load + predict + predict_one)
│   ├── data/                # loader.py, cleaning.py, split.py, encoding.py
│   ├── features/            # seasonal.py, popularity.py, customer_value.py
│   └── models/              # base.py + 4 mô hình tự cài
├── scripts/                 # explore_data, prepare_data, train, evaluate, predict_scenario
└── tests/                   # test_metrics.py, test_models.py
```

## 4. Luồng dữ liệu

```
data/raw/bigmart_sales.csv
  → loader.load_raw (kiểm tra cột, parse ngày nếu có)
  → cleaning.drop_invalid_rows (bỏ thiếu target / trùng / target < 0)
  → split.train_test_split (theo thời gian hoặc ngẫu nhiên, seed 42) → train 6819 / test 1704
  → Cleaner.fit(train) → transform train & test (median/mode, kẹp IQR, không kẹp target)
  → seasonal → popularity (product_freq) → customer_value (fit train; các nhóm không có cột thì bỏ qua)
  → bỏ DATE_COL / PRODUCT_ID_COL / CUSTOMER_ID_COL
  → CategoricalEncoder.fit(train, CATEGORICAL_COLS) → one-hot (top 20 giá trị + "__other__")
  → căn cột test theo train, target cột cuối, chỉ còn cột số (52 đặc trưng)
  → data/processed/{train,test}.csv + save_models/preprocessor.npy
  → scripts/train.py: grid search 5-fold (tuning.py) trên train → fit bản tốt nhất → save_models/<model>.npy
  → scripts/evaluate.py: đánh giá trên test → reports/results.csv + save_models/best_model.txt + hình
  → dự đoán: kịch bản (dict/JSON) → SalesPredictor (preprocessor.npy + best model) → doanh số
```

## 5. Giao diện các module chính (phải khớp với code)

- `loader.load_raw(path: Path | None = None) -> pd.DataFrame`
- `cleaning.drop_invalid_rows(df) -> pd.DataFrame`; `cleaning.Cleaner`: `fit(df) -> self`, `transform(df) -> pd.DataFrame` (chạy được khi không có target)
- `split.train_test_split(df, test_size, seed, date_col=None) -> (train_df, test_df)`
- `seasonal.add_seasonal_features(df, date_col, holiday_col=None) -> pd.DataFrame`
- `PopularityFeatures` / `CustomerValueFeatures`: `fit(train_df) -> self`, `transform(df) -> pd.DataFrame`
- `encoding.CategoricalEncoder(max_categories=20)`: `fit(df, cols) -> self`, `transform(df) -> pd.DataFrame`, có `feature_names_`
- `save_models/preprocessor.npy = {"cleaner","popularity","customer","encoder","feature_columns"}`
- `metrics`: `mse, mae, rmse, r2 (y_true, y_pred) -> float`; `evaluate_all -> dict` (khóa mse/mae/rmse/r2)
- `BaseRegressor`: `fit(X, y) -> self`, `predict(X) -> np.ndarray`, `get_params()`, `set_params(**p)`, `clone()`
- Lưu/nạp mô hình dạng **`.npy` của numpy**: `save_model(model, path)` / `load_model(path)` trong `src/models/base.py`; mỗi mô hình có `get_state()`/`from_state()`, phần trạng thái đã học chỉ gồm mảng numpy thuần (cây quyết định lưu bằng structured array, forest/boosting là một mảng có thêm cột `tree`)
- 4 mô hình: `LinearRegression(alpha)`, `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`
- `tuning.grid_search(model, param_grid, X, y, k, metric="rmse", seed=42) -> (best_params, results_df)`
- `predict.SalesPredictor.load(models_dir)`; `.predict(df_raw) -> np.ndarray`; `.predict_one(record: dict) -> float`

## 6. Kết quả đã đạt (bản đầy đủ, xem `reports/results.csv`)

| model | mse | mae | rmse | r2 |
|---|---|---|---|---|
| gradient_boosting | 1115280.05 | 740.56 | **1056.07** | **0.6118** |
| random_forest | 1135939.87 | 745.86 | 1065.80 | 0.6046 |
| decision_tree | 1152400.45 | 752.64 | 1073.50 | 0.5989 |
| linear_regression | 1236646.47 | 823.42 | 1112.05 | 0.5696 |

Mô hình tốt nhất: **gradient_boosting** (ghi trong `save_models/best_model.txt`).

## 7. Cách hoạt động end-to-end

Phần này mô tả cách cả hệ thống vận hành từ dữ liệu thô đến một dự đoán mới.

### 7.1 Các giai đoạn & lệnh chạy (từ thư mục gốc)

```bash
python -m scripts.explore_data              # 1. EDA: thống kê + 8 hình eda_*.png
python -m scripts.prepare_data              # 2. tiền xử lý → data/processed/ + preprocessor.npy
python -m scripts.train --fast              # 3a. train nhanh (tune trên 20% train) để ước lượng thời gian
python -m scripts.train                     # 3b. train đầy đủ: grid search 5-fold trên toàn bộ train
python -m scripts.evaluate                  # 4. đánh giá trên test → results.csv + best_model.txt + hình
python -m scripts.predict_scenario --demo   # 5. dự đoán kịch bản mẫu
python -m scripts.predict_scenario --input scenarios.json
python -m pytest -q                         # 6. test (17 test)
```

### 7.2 Dòng dữ liệu chi tiết

1. **`scripts/explore_data.py`** đọc `data/raw/bigmart_sales.csv` qua `loader.load_raw` (kiểm tra đủ cột bắt buộc, không sửa dữ liệu), in thống kê ra console và lưu hình vào `reports/figures/`. Kết quả EDA thật: thiếu `Item_Weight` 17.17%, `Outlet_Size` 28.28%; ngoại lai IQR ở `Item_Visibility` (144 dòng) và target (186 dòng); không có cột ngày.

2. **`scripts/prepare_data.py`** nối toàn bộ tiền xử lý theo đúng thứ tự (mọi thống kê chỉ fit trên train):
   `load_raw → drop_invalid_rows → train_test_split(ngẫu nhiên, seed 42: 6819/1704) → Cleaner.fit(train).transform cả hai → add_seasonal_features (bỏ qua vì không có ngày) → PopularityFeatures (chỉ thêm product_freq) → CustomerValueFeatures (bỏ qua, không có mã khách) → bỏ cột ID → CategoricalEncoder one-hot (top 20 + "__other__") → căn cột test theo train, target cột cuối`.
   Kết quả: `train.csv` 6819×53, `test.csv` 1704×53, không NaN, 52 đặc trưng số, và `save_models/preprocessor.npy` chứa cả bộ tiền xử lý để dùng lại khi dự đoán.

3. **`scripts/train.py`** đọc `train.csv`, với mỗi mô hình chạy `grid_search` (5-fold CV, metric RMSE, seed 42) trên lưới tham số (tối đa 8 tổ hợp/mô hình), chọn `best_params` rồi **fit lại trên toàn bộ train**, lưu `save_models/<model>.npy`, `save_models/best_params.json`, `reports/tuning_results.csv`. Cờ `--fast` chỉ dùng 20% train cho bước tuning.

4. **`scripts/evaluate.py`** nạp 4 mô hình đã lưu, tính `evaluate_all` (mse/mae/rmse/r2) trên `test.csv`, lưu `reports/results.csv` (sắp xếp theo RMSE), ghi tên mô hình tốt nhất vào `save_models/best_model.txt` và vẽ 3 hình `eval_*.png`. **Không train lại gì ở bước này.**

5. **Dự đoán mới** (`src/predict.py` + `scripts/predict_scenario.py`): `SalesPredictor.load()` nạp `preprocessor.npy` + `best_model.txt` + `<model>.npy`; `predict(df_raw)` áp lại **đúng cùng thứ tự** tiền xử lý như lúc train (Cleaner → seasonal → popularity → customer → bỏ ID → encoder → căn cột theo `feature_columns`, thiếu thì điền 0) rồi `model.predict`, kẹp kết quả ≥ 0. `--demo` dựng một dòng "điển hình" từ raw (median số / mode phân loại) làm baseline rồi tạo kịch bản (sản phẩm mới, quảng cáo +50%, ngày lễ — kịch bản nào thiếu cột thì in thông báo bỏ qua).

### 7.3 Kết quả thật khi chạy

- Test: gradient_boosting RMSE **1056.07** / R2 **0.6118** (tốt nhất), random_forest 1065.80, decision_tree 1073.50, linear_regression 1112.05.
- CV 5-fold trên train: GB 1087.02, RF 1095.56, DT 1121.37, LR 1138.49.
- Demo: baseline 3165.81; sản phẩm mới (product_freq=0) → 3411.43 (+7.76%).
- `pytest`: 17 passed.

### 7.4 Các artifact sinh ra ở đâu

| Artifact | File |
|---|---|
| Dữ liệu đã xử lý | `data/processed/train.csv`, `test.csv` |
| Bộ tiền xử lý | `save_models/preprocessor.npy` |
| 4 mô hình đã train | `save_models/{linear_regression,decision_tree,random_forest,gradient_boosting}.npy` |
| Siêu tham số tốt nhất | `save_models/best_params.json`, `reports/tuning_results.csv` |
| Mô hình được chọn | `save_models/best_model.txt` |
| Bảng đánh giá test | `reports/results.csv` |
| Hình EDA + đánh giá | `reports/figures/*.png` (8 hình EDA + 3 hình eval) |
| Báo cáo | `reports/REPORT.md` |

### 7.5 Điểm cần nhớ khi đọc code

- Không rò rỉ: `Cleaner`, `PopularityFeatures`, `CustomerValueFeatures`, `CategoricalEncoder` đều `fit` trên train rồi mới `transform` test/dữ liệu mới; `SalesPredictor` nạp nguyên bộ này từ `preprocessor.npy`.
- Mô hình tự cài kế thừa `BaseRegressor` (`fit/predict/get_params/set_params/clone`) nên `tuning.py` đối xử đồng nhất cả 4 mô hình qua `clone()` + `set_params()`.
- `DecisionTreeRegressor` tìm điểm tách O(n log n) bằng tổng tích lũy, nên Random Forest và Gradient Boosting (xây trên nó) vẫn chạy nhanh ở 8523 dòng.
- Đổi dataset khác: chỉ cần sửa tên cột trong `src/config.py` (cột không có → `None`), toàn bộ pipeline tự bỏ qua phần tương ứng.
