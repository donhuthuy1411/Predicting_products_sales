# Algorithm — tài liệu các thuật toán

Các tài liệu trong thư mục này sử dụng cùng một cấu trúc để mô tả từng thuật toán hoặc phương pháp hỗ trợ:

1. **Mục tiêu** — bài toán và vai trò của thuật toán trong dự án.
2. **Input / Output** — dữ liệu đầu vào, kết quả đầu ra, shape và tham số chính.
3. **Tiêu chí đánh giá** — metric, cách tuning và kết quả thực nghiệm.
4. **Cách hoạt động (How it works)** — các bước xử lý chính của thuật toán.
5. **Dữ liệu sử dụng (Data used)** — nguồn dữ liệu, kích thước, đặc trưng và cách dùng train/test.

| File                       | Thuật toán / phương pháp               | File cài đặt                      | Kết quả test (RMSE / R²) |
| -------------------------- | -------------------------------------- | --------------------------------- | -----------------------: |
| `linear-regression.md`     | Linear Regression (Ridge, nghiệm đóng) | `src/models/linear_regression.py` |         1112.05 / 0.5696 |
| `decision-tree.md`         | Decision Tree Regressor (CART)         | `src/models/decision_tree.py`     |         1073.50 / 0.5989 |
| `random-forest.md`         | Random Forest Regressor                | `src/models/random_forest.py`     |         1065.80 / 0.6046 |
| `gradient-boosting.md`     | Gradient Boosting Regressor            | `src/models/gradient_boosting.py` |     **1056.07 / 0.6118** |
| `k-fold-cv-grid-search.md` | K-Fold Cross-Validation + Grid Search  | `src/tuning.py`                   |                        — |

Bốn mô hình hồi quy được tự cài đặt bằng NumPy và dùng chung giao diện `BaseRegressor` tại `src/models/base.py`. K-Fold CV + Grid Search là phương pháp lựa chọn siêu tham số, không phải một mô hình dự đoán độc lập.
