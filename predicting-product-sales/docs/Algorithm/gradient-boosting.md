# Gradient Boosting Regressor (Hồi quy tăng cường Gradient)

> Mã nguồn: `src/models/gradient_boosting.py` | Thuật toán nền: `DecisionTreeRegressor` | Hàm mất mát: MSE

## 1. Mục tiêu

Xây dựng mô hình Gradient Boosting để dự đoán doanh số sản phẩm bằng cách kết hợp nhiều cây quyết định hồi quy theo phương pháp boosting.

Khác với Random Forest, các cây trong Gradient Boosting không được huấn luyện độc lập. Mỗi cây mới được xây dựng dựa trên phần sai số còn lại của mô hình trước đó, nhằm cải thiện dần kết quả dự đoán.

Mục tiêu của mô hình là giảm sai số dự đoán và nâng cao hiệu suất so với các mô hình hồi quy cơ bản.

## 2. Input / Output

|                     | Kiểu                       | Shape            | Ý nghĩa                                   |
| ------------------- | -------------------------- | ---------------- | ----------------------------------------- |
| Input `fit(X, y)`   | `np.ndarray`, `np.ndarray` | `(n, p)`, `(n,)` | Dữ liệu đặc trưng và doanh số thực tế     |
| Input `predict(X)`  | `np.ndarray`               | `(m, p)`         | Các mẫu cần dự đoán                       |
| Output `predict(X)` | `np.ndarray`               | `(m,)`           | Doanh số dự đoán                          |
| `init_`             | `float`                    | —                | Giá trị dự đoán khởi tạo                  |
| `estimators_`       | `list`                     | —                | Danh sách các cây được huấn luyện tuần tự |
| `train_loss_`       | `list[float]`              | —                | MSE trên tập train sau mỗi vòng boosting  |

Tham số chính:

- `n_estimators=100`: số lượng cây được huấn luyện.
- `learning_rate=0.1`: tốc độ đóng góp của mỗi cây vào mô hình.
- `max_depth=3`: độ sâu tối đa của mỗi cây.
- `min_samples_leaf=5`: số mẫu tối thiểu tại node lá.
- `subsample=1.0`: tỷ lệ dữ liệu được sử dụng trong mỗi vòng huấn luyện.
- `random_state=42`: kiểm soát tính ngẫu nhiên.

## 3. Công thức và tiêu chí đánh giá

Gradient Boosting sử dụng hàm mất mát bình phương:

`Loss = (1/n) * Σ(yi - F(xi))²`

Ban đầu, mô hình dự đoán bằng giá trị trung bình của biến mục tiêu:

`F0(x) = mean(y)`

Phần sai số còn lại (residual) được tính:

`ri = yi - F(xi)`

Sau khi huấn luyện cây mới, mô hình cập nhật dự đoán:

`Fm(x) = Fm-1(x) + learning_rate * hm(x)`

Trong đó:

- `hm(x)` là kết quả dự đoán của cây mới.
- `learning_rate` điều chỉnh mức đóng góp của cây mới.
- `Fm(x)` là kết quả dự đoán sau vòng huấn luyện thứ m.

Các chỉ số đánh giá:

- **MAE:** Sai số tuyệt đối trung bình.
- **MSE:** Sai số bình phương trung bình.
- **RMSE:** Căn bậc hai của MSE.
- **R²:** Đánh giá mức độ giải thích biến thiên của doanh số.

Kết quả trên tập test (`reports/results.csv`):

|    MAE |          MSE |     RMSE |     R² |
| -----: | -----------: | -------: | -----: |
| 740.56 | 1,115,280.05 | 1,056.07 | 0.6118 |

Siêu tham số được lựa chọn: `n_estimators=100`, `learning_rate=0.05`, `max_depth=3`.

Đây là mô hình có RMSE thấp nhất và R² cao nhất trong bốn mô hình được thử nghiệm.

## 4. Cách hoạt động (How it works)

1. Nhận dữ liệu huấn luyện và khởi tạo giá trị dự đoán ban đầu bằng trung bình của `y`.
2. Tính residual bằng cách lấy doanh số thực tế trừ đi kết quả dự đoán hiện tại.
3. Nếu `subsample` nhỏ hơn 1, lấy ngẫu nhiên một phần dữ liệu không hoàn lại để huấn luyện cây mới. Nếu bằng 1, sử dụng toàn bộ tập train.
4. Tạo một cây quyết định hồi quy nông và huấn luyện cây trên dữ liệu đầu vào cùng residual tương ứng.
5. Lấy kết quả dự đoán của cây mới nhân với `learning_rate`, sau đó cộng vào kết quả dự đoán hiện tại.
6. Lặp lại quá trình trên cho đến khi đạt đủ số lượng cây `n_estimators`.
7. Sau mỗi vòng, tính MSE trên tập train và lưu vào `train_loss_` để theo dõi quá trình huấn luyện.
8. Khi dự đoán dữ liệu mới, bắt đầu từ giá trị `init_`, sau đó cộng lần lượt kết quả của tất cả cây đã huấn luyện.

## 5. Dữ liệu sử dụng (Data used)

- Nguồn dữ liệu: `data/processed/train.csv` và `data/processed/test.csv`.
- Tập train gồm 6.819 mẫu, tập test gồm 1.704 mẫu.
- Mỗi mẫu có 52 đặc trưng sau tiền xử lý.
- Biến mục tiêu là `Item_Outlet_Sales`.
- Các cây được huấn luyện tuần tự dựa trên residual của mô hình hiện tại.

Với cấu hình `subsample=1.0`, mỗi cây sử dụng toàn bộ tập train. Tập test chỉ được sử dụng để đánh giá sau khi huấn luyện hoàn tất.

## 6. Yêu cầu triển khai code

- Tạo lớp `GradientBoostingRegressor` kế thừa từ `BaseRegressor`.
- Sử dụng `DecisionTreeRegressor` đã xây dựng trong dự án làm thuật toán nền.
- Không sử dụng `sklearn.ensemble.GradientBoostingRegressor`.
- Hỗ trợ các tham số `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, `subsample` và `random_state`.
- Sử dụng hàm mất mát bình phương MSE và residual `y - F` trong quá trình huấn luyện.
- Phải kiểm tra `subsample` nằm trong khoảng `(0, 1]`.
- Mỗi cây mới phải được huấn luyện dựa trên residual của mô hình hiện tại.
- Lưu giá trị khởi tạo, danh sách cây và lịch sử loss để phục vụ dự đoán và theo dõi.
- Phương thức `predict()` phải cộng kết quả của các cây với giá trị khởi tạo theo đúng thứ tự.
- Thông báo lỗi nếu mô hình chưa được huấn luyện.
- Hỗ trợ lưu và khôi phục trạng thái mô hình thông qua cơ chế của `BaseRegressor`.
