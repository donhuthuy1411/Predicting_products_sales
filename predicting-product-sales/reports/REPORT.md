# Báo cáo Practice 2 — Predicting Product Sales

## 1. Bài toán

Dự đoán doanh số sản phẩm (biến mục tiêu liên tục → **hồi quy**). Pipeline: chuẩn bị dữ liệu → tạo đặc trưng → huấn luyện/tinh chỉnh 4 mô hình (Linear Regression, Decision Tree, Random Forest, Gradient Boosting) → đánh giá bằng MSE/MAE/RMSE/R2 → "triển khai" bằng class `SalesPredictor` và CLI dự đoán theo kịch bản (`scripts/predict_scenario.py`).

Ràng buộc quan trọng: **mô hình tự cài đặt bằng numpy**, không dùng scikit-learn trong `src/` và `scripts/`.

## 2. Dữ liệu

- **Dataset**: BigMart Sales (Kaggle — "Big Mart Sales Prediction", file gốc `Train.csv` đổi tên thành `bigmart_sales.csv`), đặt tại `data/raw/bigmart_sales.csv`.
- **Kích thước**: 8523 dòng × 12 cột. Biến mục tiêu: `Item_Outlet_Sales` (doanh số sản phẩm tại outlet).
- **Cột số**: `Item_Weight`, `Item_Visibility`, `Item_MRP`, `Outlet_Establishment_Year`.
- **Cột phân loại**: `Item_Fat_Content`, `Item_Type`, `Outlet_Identifier`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`.
- **Cột tùy chọn không có trong bộ này** (đặt `None` trong `src/config.py`): ngày (`DATE_COL`), ngày lễ, mã khách hàng, số lượng, số review, rating, chi phí quảng cáo. Riêng `PRODUCT_ID_COL = Item_Identifier` và `PRICE_COL = Item_MRP` có sẵn.

Kết quả EDA (`python -m scripts.explore_data`, hình trong `reports/figures/`):

- Thiếu giá trị: `Item_Weight` 17.17%, `Outlet_Size` 28.28%; các cột còn lại không thiếu. Không có dòng trùng; không thiếu target.
- Ngoại lai theo IQR (k=1.5): `Item_Outlet_Sales` 186 dòng (target — **không kẹp**), `Item_Visibility` 144 dòng.
- Không có cột ngày → chia train/test **ngẫu nhiên** (seed 42): train 6819 dòng, test 1704 dòng.

## 3. Tiền xử lý

- `drop_invalid_rows`: bỏ dòng thiếu target, bỏ dòng trùng, bỏ dòng target < 0 (stateless).
- `Cleaner` (fit trên train): điền thiếu bằng median (số) / mode (phân loại); kẹp (winsorize) ngoại lai của cột số theo cận IQR `[Q1−1.5·IQR, Q3+1.5·IQR]` đã học, **không kẹp target**.
- Chia train/test: không có cột ngày nên xáo trộn bằng `np.random.default_rng(42)` rồi chia 80/20.
- One-hot (`CategoricalEncoder`): giữ tối đa 20 giá trị phổ biến nhất mỗi cột, phần còn lại gom về `__other__`; giá trị lạ ở test → `__other__`. Fit trên train, áp cho test.
- Chống rò rỉ: mọi thống kê (median, mode, cận IQR, tần suất sản phẩm, bảng mã hóa) chỉ học trên tập train.
- Kết quả (`python -m scripts.prepare_data`): `data/processed/train.csv` (6819×53), `test.csv` (1704×53), không còn NaN, target ở cột cuối, chỉ còn cột số, **52 đặc trưng**; lưu `save_models/preprocessor.npy`.

## 4. Đặc trưng

- **Mùa vụ/lễ** (`seasonal.py`): dataset không có cột ngày → hàm trả dữ liệu nguyên vẹn (bỏ qua, không lỗi).
- **Độ phổ biến sản phẩm** (`popularity.py`): thêm `product_freq` = số lần xuất hiện của `Item_Identifier` trong train; sản phẩm mới → 0. (Không có review/rating nên không tạo `log_reviews`, `rating_x_reviews`.)
- **Giá trị khách hàng (CLV)** (`customer_value.py`): không có `CUSTOMER_ID_COL` → bỏ qua nhóm đặc trưng này.
- Sau đó bỏ các cột ngày/ID (`DATE_COL`, `PRODUCT_ID_COL`, `CUSTOMER_ID_COL`) trước khi vào mô hình.

## 5. Mô hình & siêu tham số

Tất cả tự cài bằng numpy, kế thừa `BaseRegressor` (`fit/predict/get_params/set_params/clone`):

| Mô hình | Ý tưởng | Lưới tinh chỉnh (grid search, 5-fold CV, metric RMSE) | best_params |
|---|---|---|---|
| Linear Regression | ridge, nghiệm đóng, chuẩn hóa đặc trưng, không phạt intercept | alpha ∈ {0, 1, 10, 100} | alpha=10 |
| Decision Tree | CART hồi quy, tìm điểm tách giảm MSE bằng tổng tích lũy O(n log n) | max_depth ∈ {4,8,12}, min_samples_leaf ∈ {5,20} | max_depth=4, min_samples_leaf=20 |
| Random Forest | bagging, mỗi cây trên một mẫu bootstrap, chọn ngẫu nhiên đặc trưng ("third") | n_estimators ∈ {30,60}, max_depth ∈ {8,12} | n_estimators=60, max_depth=12 |
| Gradient Boosting | học trên residual với loss MSE, cây nông depth 3 | n_estimators ∈ {50,100}, learning_rate ∈ {0.05,0.1}, max_depth=3 | n_estimators=100, learning_rate=0.05, max_depth=3 |

Kết quả 5-fold CV trên train (đầy đủ, xem `reports/tuning_results.csv`): RMSE lần lượt 1138.49 / 1121.37 / 1095.56 / **1087.02** (gradient_boosting).

## 6. Kết quả trên tập test

`python -m scripts.evaluate` → `reports/results.csv` (sắp xếp tăng dần theo RMSE):

| model | mse | mae | rmse | r2 |
|---|---|---|---|---|
| **gradient_boosting** | 1115280.05 | 740.56 | **1056.07** | **0.6118** |
| random_forest | 1135939.87 | 745.86 | 1065.80 | 0.6046 |
| decision_tree | 1152400.45 | 752.64 | 1073.50 | 0.5989 |
| linear_regression | 1236646.47 | 823.42 | 1112.05 | 0.5696 |

Mô hình tốt nhất (RMSE nhỏ nhất): **gradient_boosting** → ghi vào `save_models/best_model.txt`.

Hình: `reports/figures/eval_rmse_bar.png`, `eval_pred_vs_actual.png` (đường y=x), `eval_residual_hist.png`, cùng các hình EDA (`eda_*.png`).

## 7. Phân tích

- Gradient Boosting tốt nhất (RMSE 1056.07, R2 0.6118), theo sau là Random Forest — các mô hình phi tuyến tận dụng tốt quan hệ giữa `Item_MRP`, `Item_Visibility` và các đặc trưng one-hot của outlet với doanh số; Linear Regression kém nhất vì quan hệ phi tuyến (xem heatmap tương quan ở `eda_corr_heatmap.png`).
- Decision Tree đơn lẻ overfit khi depth lớn (CV xấu đi ở depth 12), nên grid search chọn depth 4.
- Hạn chế: bộ dữ liệu không có cột thời gian, khách hàng, review/rating, chi phí quảng cáo nên một số nhóm đặc trưng và kịch bản bị bỏ qua; R2 ~0.61 cho thấy vẫn còn nhiều phương sai chưa giải thích được (ví dụ: tương tác giữa loại sản phẩm và loại cửa hàng).

## 8. Triển khai

- `SalesPredictor.load()` nạp `save_models/preprocessor.npy` + `save_models/best_model.txt` + `save_models/<model>.npy`; `predict(df_raw)` áp đúng thứ tự pipeline huấn luyện (Cleaner → seasonal → popularity → customer → bỏ cột ID → encoder → căn cột theo `feature_columns`) rồi dự đoán, kẹp kết quả ≥ 0; `predict_one(dict)` cho 1 bản ghi.
- CLI kịch bản: `python -m scripts.predict_scenario --demo` (hoặc `--input scenarios.json`). Với bộ BigMart: kịch bản "sản phẩm mới" (product_freq=0) dự đoán 3411.43 so với baseline 3165.81 (**+7.76%**); các kịch bản quảng cáo/ngày lễ bị bỏ qua vì dataset không có các cột đó (in thông báo, không lỗi).

## 9. Hướng cải tiến

- ARIMA/Prophet cho chuỗi thời gian nếu có cột ngày (bộ này không có → ngoài phạm vi, chỉ nhắc ở đây).
- Hệ gợi ý (recommendation system) — ngoài phạm vi đề.
- Thêm đặc trưng: tương tác giá × loại sản phẩm, nhóm theo `Outlet_Establishment_Year`, xử lý chuẩn hơn `Item_Fat_Content` (các giá trị "LF"/"low fat"/"reg" chưa đồng nhất); thử stacking các mô hình; dùng dataset có thêm review/rating/chi phí quảng cáo để đủ các nhóm đặc trưng và kịch bản còn thiếu.
