# Linear Regression (Ridge Regression – nghiệm đóng)

> Mã nguồn: `src/models/linear_regression.py` | Mô hình: `save_models/linear_regression.npy` | Đánh giá: `reports/results.csv`

## 1. Mục tiêu

Xây dựng mô hình hồi quy tuyến tính làm baseline để dự đoán doanh số `Item_Outlet_Sales` từ các đặc trưng sản phẩm và cửa hàng. Mô hình sử dụng regularization L2 (Ridge) nhằm hạn chế overfitting khi có nhiều đặc trưng.

## 2. Input / Output

|                              | Kiểu                       | Shape            | Ý nghĩa                                       |
| ---------------------------- | -------------------------- | ---------------- | --------------------------------------------- |
| Input `fit(X, y)`            | `np.ndarray`, `np.ndarray` | `(n, p)`, `(n,)` | `X`: ma trận đặc trưng; `y`: doanh số thực tế |
| Output `predict(X)`          | `np.ndarray`               | `(n,)`           | Doanh số dự đoán cho `n` mẫu                  |
| Model attribute `coef_`      | `np.ndarray`               | `(p,)`           | Hệ số của các đặc trưng trên thang đo gốc     |
| Model attribute `intercept_` | `float`                    | —                | Hệ số chặn trên thang đo gốc                  |

Tham số: `alpha=0.0` là hệ số phạt L2. Trong lần tuning hiện tại, `alpha=10` được lựa chọn. Gọi `predict` trước `fit` sẽ phát sinh `RuntimeError`.

## 3. Tiêu chí đánh giá (evaluation metric / criteria)

- **MAE**: sai số tuyệt đối trung bình giữa doanh số thực tế và dự đoán.
- **MSE**: trung bình bình phương sai số.
- **RMSE**: căn bậc hai của MSE; càng thấp càng tốt.
- **R²**: mức độ mô hình giải thích biến thiên của doanh số; càng cao càng tốt.
- Tuning dùng 5-fold Cross Validation với RMSE; lưới `alpha ∈ {0, 1, 10, 100}`.

Kết quả trên tập test (`reports/results.csv`):

|          MSE |    MAE |     RMSE |     R² |
| -----------: | -----: | -------: | -----: |
| 1,236,646.47 | 823.42 | 1,112.05 | 0.5696 |

Siêu tham số được chọn: `alpha=10` (CV RMSE = 1138.49). Đây là kết quả thấp hơn về hiệu suất so với ba mô hình còn lại trong lần thực nghiệm hiện tại.

## 4. Cách hoạt động (How it works)

1. **Chuẩn hóa đặc trưng**: tính và lưu `x_mean_`, `x_std_` từ tập train; nếu một cột có độ lệch chuẩn bằng 0 thì thay bằng 1. Tạo `Xs = (X - mean) / std`.
2. **Thêm bias**: ghép một cột 1 vào `Xs` để tạo ma trận `Xb = [1, Xs]`.
3. **Tìm nghiệm Ridge**: giải `(Xb.T @ Xb + αI)w = Xb.T @ y` bằng `np.linalg.solve`; nếu ma trận suy biến thì dùng `np.linalg.lstsq`. Phần tử phạt tại vị trí intercept được đặt bằng 0 nên không regularize hệ số chặn.
4. **Đưa hệ số về thang đo gốc**: tính `coef_ = w[1:] / std` và `intercept_ = w[0] - coef_ @ mean`.
5. **Dự đoán**: áp dụng phép chuẩn hóa đã lưu cho dữ liệu đầu vào rồi tính `ŷ = Xb @ w`.

Công thức tối ưu: `argmin_w ||Xb @ w - y||² + alpha * ||w[1:]||²`.

## 5. Dữ liệu sử dụng (Data used)

- Nguồn: `data/processed/train.csv` và `data/processed/test.csv`.
- Tập train gồm 6.819 mẫu, tập test gồm 1.704 mẫu; mỗi mẫu có 52 đặc trưng và target `Item_Outlet_Sales`.
- Đặc trưng gồm các cột số, `product_freq` và các cột One-Hot Encoding được tạo trong bước tiền xử lý.
- Chỉ dùng tập train để tính thống kê chuẩn hóa và fit mô hình; tập test chỉ dùng để đánh giá.
