# Linear Regression (Hồi quy tuyến tính – Ridge Regression)

> Mã nguồn: `src/models/linear_regression.py` | Lớp cơ sở: `src/models/base.py` | Mô hình lưu: `.npy`

## 1. Mục tiêu

Xây dựng mô hình hồi quy tuyến tính để dự đoán doanh số sản phẩm `Item_Outlet_Sales` dựa trên các đặc trưng của sản phẩm và cửa hàng.

Mô hình tìm mối quan hệ tuyến tính giữa các biến đầu vào và doanh số đầu ra. Ngoài ra, sử dụng Ridge Regularization (L2) để hạn chế các hệ số trở nên quá lớn, giúp mô hình ổn định hơn khi có nhiều đặc trưng.

Trong dự án, Linear Regression được sử dụng làm mô hình cơ sở để so sánh với Decision Tree, Random Forest và Gradient Boosting.

## 2. Input / Output

|                     | Kiểu                       | Shape            | Ý nghĩa                                           |
| ------------------- | -------------------------- | ---------------- | ------------------------------------------------- |
| Input `fit(X, y)`   | `np.ndarray`, `np.ndarray` | `(n, p)`, `(n,)` | `X` là ma trận đặc trưng, `y` là doanh số thực tế |
| Input `predict(X)`  | `np.ndarray`               | `(m, p)`         | Dữ liệu cần dự đoán                               |
| Output `predict(X)` | `np.ndarray`               | `(m,)`           | Doanh số dự đoán của từng mẫu                     |
| `coef_`             | `np.ndarray`               | `(p,)`           | Các hệ số tương ứng với đặc trưng                 |
| `intercept_`        | `float`                    | —                | Hệ số chặn của mô hình                            |

Trong đó:

- `n`: số mẫu dùng để huấn luyện.
- `m`: số mẫu cần dự đoán.
- `p`: số đặc trưng đầu vào.

Tham số chính:

- `alpha=0.0`: hệ số điều chỉnh mức phạt L2. Alpha càng lớn thì mức regularization càng mạnh.

## 3. Công thức và tiêu chí đánh giá

Mô hình hồi quy tuyến tính có dạng:

`y_pred = w0 + w1*x1 + w2*x2 + ... + wp*xp`

Trong đó:

- `w0`: hệ số chặn.
- `w1, w2, ..., wp`: trọng số của các đặc trưng.
- `x1, x2, ..., xp`: các giá trị đầu vào.

Hàm mục tiêu Ridge Regression:

`Loss = ||Xw - y||² + alpha * ||w||²`

Hệ số chặn không bị áp dụng regularization.

Các chỉ số đánh giá:

- **MAE:** Sai số tuyệt đối trung bình.
- **MSE:** Sai số bình phương trung bình.
- **RMSE:** Căn bậc hai của MSE.
- **R²:** Đánh giá mức độ mô hình giải thích sự biến thiên của doanh số.

Kết quả trên tập test (`reports/results.csv`):

|    MAE |          MSE |     RMSE |     R² |
| -----: | -----------: | -------: | -----: |
| 823.42 | 1,236,646.47 | 1,112.05 | 0.5696 |

Siêu tham số được lựa chọn trong quá trình tuning: `alpha=10`.

## 4. Cách hoạt động (How it works)

1. Nhận dữ liệu đầu vào `X` và nhãn thực tế `y`, kiểm tra số dòng giữa hai dữ liệu phải bằng nhau.
2. Tính giá trị trung bình và độ lệch chuẩn của từng đặc trưng trên tập train để chuẩn hóa dữ liệu. Nếu một cột có độ lệch chuẩn bằng 0 thì thay bằng 1.
3. Thêm một cột có giá trị 1 vào ma trận đặc trưng để biểu diễn hệ số chặn.
4. Tính nghiệm của bài toán Ridge bằng phương pháp đại số tuyến tính. Nếu phép giải ma trận không thành công thì sử dụng phương pháp least squares dự phòng.
5. Lưu các trọng số, trung bình và độ lệch chuẩn để sử dụng khi dự đoán.
6. Khi nhận dữ liệu mới, thực hiện chuẩn hóa bằng các giá trị đã lưu, sau đó tính kết quả dự đoán.

## 5. Dữ liệu sử dụng (Data used)

- Dữ liệu được lấy từ `data/processed/train.csv` và `data/processed/test.csv`.
- Tập train có 6.819 mẫu, tập test có 1.704 mẫu.
- Mỗi mẫu gồm 52 đặc trưng sau tiền xử lý.
- Biến mục tiêu là `Item_Outlet_Sales`.
- Các đặc trưng đầu vào gồm dữ liệu số, đặc trưng được tạo thêm và các cột đã mã hóa One-Hot Encoding.

Chỉ sử dụng tập train để tính các thống kê chuẩn hóa và huấn luyện mô hình. Tập test được giữ riêng để đánh giá.

## 6. Yêu cầu triển khai code

- Tạo lớp `LinearRegression` kế thừa từ `BaseRegressor`.
- Sử dụng NumPy để thực hiện các phép tính ma trận.
- Không sử dụng `sklearn.linear_model` hoặc thư viện có sẵn để thay thế thuật toán.
- Cần có hai phương thức chính `fit(X, y)` và `predict(X)`.
- Lưu các thuộc tính đã học gồm `w_`, `x_mean_`, `x_std_`, `coef_` và `intercept_`.
- Kiểm tra dữ liệu đầu vào, báo lỗi nếu `X` không phải ma trận 2 chiều hoặc số mẫu không khớp với `y`.
- Nếu gọi `predict()` trước khi `fit()` thì phải thông báo lỗi rõ ràng.
- Hỗ trợ `get_params()`, `set_params()`, `clone()` và cơ chế lưu/nạp mô hình thông qua `BaseRegressor`.
- Kết quả dự đoán phải là mảng NumPy một chiều.
