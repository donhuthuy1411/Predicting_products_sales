# Random Forest Regressor

> Mã nguồn: `src/models/random_forest.py` | Thành phần nền: `DecisionTreeRegressor` | Đánh giá: `reports/results.csv`

## 1. Mục tiêu

Giảm phương sai của một cây quyết định đơn bằng phương pháp bagging. Random Forest huấn luyện nhiều cây trên các mẫu bootstrap khác nhau, đồng thời giới hạn tập đặc trưng được xem xét tại mỗi node, sau đó lấy trung bình kết quả dự đoán của các cây.

## 2. Input / Output

|                               | Kiểu                          | Shape            | Ý nghĩa                                      |
| ----------------------------- | ----------------------------- | ---------------- | -------------------------------------------- |
| Input `fit(X, y)`             | `np.ndarray`, `np.ndarray`    | `(n, p)`, `(n,)` | Ma trận đặc trưng và vector doanh số thực tế |
| Output `predict(X)`           | `np.ndarray`                  | `(n,)`           | Trung bình dự đoán của tất cả cây trong rừng |
| Model attribute `estimators_` | `list[DecisionTreeRegressor]` | —                | Danh sách các cây đã được huấn luyện         |

Tham số mặc định: `n_estimators=50`, `max_depth=10`, `min_samples_leaf=5`, `max_features="third"`, `random_state=42`. Gọi `predict` trước `fit` sẽ phát sinh `RuntimeError`.

## 3. Tiêu chí đánh giá (evaluation metric / criteria)

- **MAE**: sai số tuyệt đối trung bình.
- **MSE**: trung bình bình phương sai số.
- **RMSE**: căn bậc hai của MSE; càng thấp càng tốt.
- **R²**: hệ số xác định; càng cao càng tốt.
- Tuning dùng 5-fold Cross Validation với lưới `n_estimators ∈ {30, 60}` và `max_depth ∈ {8, 12}`.

Kết quả trên tập test (`reports/results.csv`):

|          MSE |    MAE |     RMSE |     R² |
| -----------: | -----: | -------: | -----: |
| 1,135,939.87 | 745.86 | 1,065.80 | 0.6046 |

Siêu tham số được chọn: `n_estimators=60`, `max_depth=12` (CV RMSE = 1095.56).

## 4. Cách hoạt động (How it works)

1. **Khởi tạo bộ sinh số ngẫu nhiên**: tạo `rng = np.random.default_rng(random_state)` để quá trình lấy mẫu có thể tái lập.
2. **Tạo mẫu bootstrap**: với mỗi cây, lấy ngẫu nhiên `n` chỉ số từ `0` đến `n-1` có hoàn lại. Mỗi cây có thể nhận một tập mẫu huấn luyện khác nhau.
3. **Huấn luyện cây**: tạo một `DecisionTreeRegressor` với các tham số đã cấu hình và `seed` riêng, sau đó fit trên mẫu bootstrap tương ứng.
4. **Lưu cây**: thêm từng cây đã huấn luyện vào danh sách `estimators_`.
5. **Dự đoán**: chạy dữ liệu đầu vào qua tất cả cây và lấy trung bình: `ŷ = (1/B) * Σ tree_b.predict(X)`, với `B` là số cây.

Hai nguồn ngẫu nhiên gồm lấy mẫu bootstrap và chọn đặc trưng ngẫu nhiên tại mỗi node giúp các cây ít tương quan với nhau hơn, từ đó giảm phương sai của mô hình ensemble.

## 5. Dữ liệu sử dụng (Data used)

- Nguồn: `data/processed/train.csv` để huấn luyện và `data/processed/test.csv` để đánh giá.
- Tập train gồm 6.819 mẫu; tập test gồm 1.704 mẫu. Mỗi mẫu có 52 đặc trưng sau tiền xử lý; target là `Item_Outlet_Sales`.
- Mỗi cây được huấn luyện trên một mẫu bootstrap có kích thước bằng tập train. Các cây sử dụng các đặc trưng đầu vào đã xử lý; số đặc trưng được xem xét tại mỗi node phụ thuộc `max_features`.
