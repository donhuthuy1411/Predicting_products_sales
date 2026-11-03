# Random Forest Regressor (Rừng ngẫu nhiên hồi quy)

> Mã nguồn: `src/models/random_forest.py` | Thuật toán nền: `DecisionTreeRegressor` | Phương pháp: Bagging

## 1. Mục tiêu

Xây dựng mô hình Random Forest để dự đoán doanh số sản phẩm bằng cách kết hợp nhiều cây quyết định hồi quy.

Một cây quyết định riêng lẻ có thể bị ảnh hưởng nhiều bởi dữ liệu huấn luyện. Random Forest khắc phục vấn đề này bằng cách tạo ra nhiều tập dữ liệu bootstrap khác nhau, huấn luyện nhiều cây và kết hợp kết quả dự đoán của chúng.

Mục tiêu là giảm sự biến động của mô hình và cải thiện khả năng dự đoán so với một cây quyết định đơn lẻ.

## 2. Input / Output

|                     | Kiểu                       | Shape            | Ý nghĩa                                         |
| ------------------- | -------------------------- | ---------------- | ----------------------------------------------- |
| Input `fit(X, y)`   | `np.ndarray`, `np.ndarray` | `(n, p)`, `(n,)` | Dữ liệu đặc trưng và doanh số thực tế           |
| Input `predict(X)`  | `np.ndarray`               | `(m, p)`         | Các mẫu cần dự đoán                             |
| Output `predict(X)` | `np.ndarray`               | `(m,)`           | Doanh số dự đoán sau khi lấy trung bình các cây |
| `estimators_`       | `list`                     | —                | Danh sách các cây đã huấn luyện                 |

Tham số chính:

- `n_estimators=50`: số lượng cây trong rừng.
- `max_depth=10`: độ sâu tối đa của mỗi cây.
- `min_samples_leaf=5`: số mẫu tối thiểu tại node lá.
- `max_features="third"`: số đặc trưng được xem xét khi chia node.
- `random_state=42`: kiểm soát quá trình lấy mẫu và tính ngẫu nhiên.

## 3. Công thức và tiêu chí đánh giá

Mỗi cây trong Random Forest tạo ra một kết quả dự đoán riêng. Kết quả cuối cùng được tính bằng trung bình dự đoán của tất cả các cây:

`y_pred = (1 / B) * Σ tree_b(X)`

Trong đó:

- `B` là tổng số cây.
- `tree_b(X)` là kết quả dự đoán của cây thứ `b`.

Các chỉ số đánh giá:

- **MAE:** Sai số tuyệt đối trung bình.
- **MSE:** Sai số bình phương trung bình.
- **RMSE:** Căn bậc hai của MSE.
- **R²:** Đánh giá mức độ giải thích biến thiên của doanh số.

Kết quả trên tập test (`reports/results.csv`):

|    MAE |          MSE |     RMSE |     R² |
| -----: | -----------: | -------: | -----: |
| 745.86 | 1,135,939.87 | 1,065.80 | 0.6046 |

Siêu tham số được lựa chọn: `n_estimators=60`, `max_depth=12`.

## 4. Cách hoạt động (How it works)

1. Nhận dữ liệu huấn luyện và khởi tạo bộ sinh số ngẫu nhiên dựa trên `random_state`.
2. Tạo nhiều tập dữ liệu bootstrap bằng cách lấy ngẫu nhiên các mẫu từ tập train có hoàn lại. Mỗi tập bootstrap có số mẫu bằng tập train ban đầu.
3. Với mỗi tập bootstrap, khởi tạo một `DecisionTreeRegressor` với các tham số đã thiết lập.
4. Huấn luyện từng cây trên tập bootstrap tương ứng. Quá trình lựa chọn đặc trưng tại mỗi node cũng có yếu tố ngẫu nhiên thông qua `max_features`.
5. Lưu tất cả các cây đã huấn luyện vào danh sách `estimators_`.
6. Khi dự đoán, đưa dữ liệu đầu vào qua từng cây, thu thập kết quả và tính trung bình để tạo ra kết quả cuối cùng.

## 5. Dữ liệu sử dụng (Data used)

- Nguồn dữ liệu: `data/processed/train.csv` và `data/processed/test.csv`.
- Tập train gồm 6.819 mẫu, tập test gồm 1.704 mẫu.
- Mỗi mẫu có 52 đặc trưng sau tiền xử lý.
- Biến mục tiêu là `Item_Outlet_Sales`.
- Mỗi cây được huấn luyện từ một mẫu bootstrap lấy từ tập train.

Tập test không tham gia vào quá trình xây dựng cây hoặc lấy mẫu bootstrap. Tập này chỉ được dùng để đánh giá kết quả cuối cùng.

## 6. Yêu cầu triển khai code

- Tạo lớp `RandomForestRegressor` kế thừa từ `BaseRegressor`.
- Sử dụng `DecisionTreeRegressor` đã xây dựng trong dự án làm thuật toán nền.
- Không sử dụng `sklearn.ensemble.RandomForestRegressor`.
- Hỗ trợ các tham số `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` và `random_state`.
- Mỗi cây phải được huấn luyện trên một tập bootstrap có hoàn lại và có kích thước bằng tập train.
- Các cây cần có seed riêng để quá trình huấn luyện có thể tái lập.
- Phương thức `fit()` phải tạo và lưu danh sách cây trong `estimators_`.
- Phương thức `predict()` phải lấy trung bình kết quả dự đoán của tất cả cây.
- Thông báo lỗi rõ ràng nếu gọi `predict()` trước khi huấn luyện.
- Hỗ trợ lưu và khôi phục trạng thái của toàn bộ các cây thông qua cơ chế của `BaseRegressor`.
