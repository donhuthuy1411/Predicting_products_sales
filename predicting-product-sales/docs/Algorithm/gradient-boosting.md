# Gradient Boosting Regressor

> Mã nguồn: `src/models/gradient_boosting.py` | Thành phần nền: `DecisionTreeRegressor` | Đánh giá: `reports/results.csv`

## 1. Mục tiêu

Xây dựng mô hình cộng nhiều cây hồi quy nông theo phương pháp boosting. Ở mỗi vòng lặp, một cây mới học trên phần dư (residual) của mô hình hiện tại nhằm giảm sai số dự đoán. Mô hình sử dụng loss bình phương (MSE).

## 2. Input / Output

|                               | Kiểu                          | Shape            | Ý nghĩa                                      |
| ----------------------------- | ----------------------------- | ---------------- | -------------------------------------------- |
| Input `fit(X, y)`             | `np.ndarray`, `np.ndarray`    | `(n, p)`, `(n,)` | Ma trận đặc trưng và vector doanh số thực tế |
| Output `predict(X)`           | `np.ndarray`                  | `(n,)`           | Doanh số dự đoán từ tổng các cây             |
| Model attribute `init_`       | `float`                       | —                | Giá trị dự đoán khởi tạo `F0`                |
| Model attribute `estimators_` | `list[DecisionTreeRegressor]` | —                | Các cây được huấn luyện tuần tự              |
| Model attribute `train_loss_` | `list[float]`                 | —                | MSE trên train sau mỗi vòng boosting         |

Tham số mặc định: `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `min_samples_leaf=5`, `subsample=1.0`, `random_state=42`. Gọi `predict` trước `fit` sẽ phát sinh `RuntimeError`.

## 3. Tiêu chí đánh giá (evaluation metric / criteria)

- **MAE**: sai số tuyệt đối trung bình.
- **MSE**: trung bình bình phương sai số; cũng là loss được tối ưu trong quá trình boosting.
- **RMSE**: căn bậc hai của MSE; metric chính để tuning và so sánh.
- **R²**: hệ số xác định.
- Tuning dùng 5-fold Cross Validation với lưới `n_estimators ∈ {50, 100}`, `learning_rate ∈ {0.05, 0.1}` và `max_depth=3`.

Kết quả trên tập test (`reports/results.csv`):

|          MSE |    MAE |     RMSE |     R² |
| -----------: | -----: | -------: | -----: |
| 1,115,280.05 | 740.56 | 1,056.07 | 0.6118 |

Siêu tham số được chọn: `n_estimators=100`, `learning_rate=0.05`, `max_depth=3` (CV RMSE = 1087.02). Đây là mô hình có RMSE thấp nhất trong bốn mô hình ở lần thực nghiệm hiện tại.

## 4. Cách hoạt động (How it works)

Với loss bình phương, gradient âm của loss tương ứng với residual `r = y - F(x)`.

1. **Khởi tạo dự đoán**: tính `F0 = mean(y)` và gán `F(x) = F0` cho tất cả mẫu.
2. **Tính residual**: ở mỗi vòng `m`, tính `r = y - F` để xác định phần sai số hiện tại.
3. **Lấy mẫu**: nếu `subsample < 1`, chọn ngẫu nhiên một phần dữ liệu không hoàn lại; nếu `subsample=1`, sử dụng toàn bộ tập train.
4. **Huấn luyện cây mới**: fit một cây nông trên dữ liệu đầu vào và residual tương ứng.
5. **Cập nhật dự đoán**: cộng dự đoán của cây mới vào mô hình hiện tại với hệ số `learning_rate`: `F += learning_rate * tree.predict(X)`.
6. **Theo dõi loss**: tính MSE trên train sau mỗi vòng và lưu vào `train_loss_`.
7. **Dự đoán**: tính `ŷ = F0 + learning_rate * Σ tree_m.predict(X)` trên các cây đã huấn luyện.

`learning_rate` điều chỉnh mức đóng góp của từng cây. Giá trị nhỏ hơn làm mỗi bước cập nhật thận trọng hơn và thường cần nhiều cây hơn.

## 5. Dữ liệu sử dụng (Data used)

- Nguồn: `data/processed/train.csv` để huấn luyện và `data/processed/test.csv` để đánh giá.
- Tập train gồm 6.819 mẫu; tập test gồm 1.704 mẫu. Mỗi mẫu có 52 đặc trưng sau tiền xử lý; target là `Item_Outlet_Sales`.
- Các cây học residual được tính từ tập train; trong cấu hình kết quả hiện tại `subsample=1.0`, do đó mỗi cây sử dụng toàn bộ tập train.
- Tập test chỉ được sử dụng để đánh giá mô hình sau huấn luyện.
