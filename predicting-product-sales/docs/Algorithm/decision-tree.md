# Decision Tree Regressor (Cây quyết định hồi quy – CART)

> Mã nguồn: `src/models/decision_tree.py` | Lớp cơ sở: `src/models/base.py` | Thuật toán: CART

## 1. Mục tiêu

Xây dựng mô hình cây quyết định hồi quy để dự đoán doanh số sản phẩm dựa trên các đặc trưng đầu vào.

Thay vì tìm một công thức tuyến tính như Linear Regression, Decision Tree chia dữ liệu thành nhiều nhóm nhỏ dựa trên các điều kiện của đặc trưng. Mỗi nhóm tương ứng với một nút lá và sử dụng giá trị trung bình doanh số của nhóm đó làm kết quả dự đoán.

Mô hình được sử dụng để tìm hiểu các mối quan hệ phi tuyến giữa đặc trưng sản phẩm, cửa hàng và doanh số.

## 2. Input / Output

|                     | Kiểu                       | Shape            | Ý nghĩa                               |
| ------------------- | -------------------------- | ---------------- | ------------------------------------- |
| Input `fit(X, y)`   | `np.ndarray`, `np.ndarray` | `(n, p)`, `(n,)` | Dữ liệu đặc trưng và doanh số thực tế |
| Input `predict(X)`  | `np.ndarray`               | `(m, p)`         | Các mẫu cần dự đoán                   |
| Output `predict(X)` | `np.ndarray`               | `(m,)`           | Doanh số dự đoán                      |
| `nodes_`            | `list[dict]`               | —                | Danh sách các node trong cây          |
| `n_features_in_`    | `int`                      | —                | Số lượng đặc trưng khi huấn luyện     |

Tham số chính:

- `max_depth=8`: độ sâu tối đa của cây.
- `min_samples_split=10`: số mẫu tối thiểu để tiếp tục chia một node.
- `min_samples_leaf=5`: số mẫu tối thiểu trong mỗi node lá.
- `max_features=None`: số lượng đặc trưng được xem xét khi tìm điểm chia.
- `random_state=None`: kiểm soát tính ngẫu nhiên khi lựa chọn đặc trưng.

## 3. Công thức và tiêu chí đánh giá

Decision Tree sử dụng tiêu chí giảm sai số bình phương để tìm cách chia dữ liệu tốt nhất.

Với một node chứa `n` mẫu, sai số bình phương được tính:

`SSE = Σ(yi - y_mean)²`

Khi chia node thành hai nhánh trái và phải, thuật toán lựa chọn điểm chia có tổng sai số nhỏ nhất:

`SSE_total = SSE_left + SSE_right`

Giá trị dự đoán tại node lá là:

`y_pred = mean(y_leaf)`

Các chỉ số đánh giá:

- **MAE:** Sai số tuyệt đối trung bình.
- **MSE:** Sai số bình phương trung bình.
- **RMSE:** Căn bậc hai của MSE.
- **R²:** Đánh giá mức độ giải thích biến thiên của doanh số.

Kết quả trên tập test (`reports/results.csv`):

|    MAE |          MSE |     RMSE |     R² |
| -----: | -----------: | -------: | -----: |
| 752.64 | 1,152,400.45 | 1,073.50 | 0.5989 |

Siêu tham số được lựa chọn: `max_depth=4`, `min_samples_leaf=20`.

## 4. Cách hoạt động (How it works)

1. Bắt đầu với toàn bộ dữ liệu huấn luyện tại node gốc.
2. Kiểm tra điều kiện dừng như đạt độ sâu tối đa, số mẫu không đủ để chia hoặc tất cả giá trị mục tiêu đều giống nhau.
3. Nếu chưa dừng, thuật toán lần lượt xét các đặc trưng và tìm những ngưỡng có thể chia dữ liệu thành hai nhánh.
4. Sắp xếp giá trị của từng đặc trưng, sau đó sử dụng tổng tích lũy để tính nhanh sai số của các cách chia hợp lệ.
5. Chọn đặc trưng và ngưỡng chia có tổng SSE nhỏ nhất.
6. Tiếp tục chia nhánh trái và nhánh phải cho đến khi thỏa mãn điều kiện dừng. Nếu không tìm được điểm chia hợp lệ thì tạo node lá.
7. Khi dự đoán, mỗi mẫu được đưa qua các node theo điều kiện so sánh với ngưỡng. Khi đến node lá, mô hình trả về giá trị trung bình được lưu tại node đó.

## 5. Dữ liệu sử dụng (Data used)

- Nguồn dữ liệu: `data/processed/train.csv` và `data/processed/test.csv`.
- Tập train gồm 6.819 mẫu, tập test gồm 1.704 mẫu.
- Mỗi mẫu có 52 đặc trưng sau bước tiền xử lý.
- Biến mục tiêu là `Item_Outlet_Sales`.
- Các đặc trưng bao gồm thông tin sản phẩm, cửa hàng và các cột được mã hóa One-Hot Encoding.

Tập train được dùng để xây dựng cây. Tập test chỉ được dùng để đánh giá khả năng dự đoán trên dữ liệu chưa được mô hình nhìn thấy.

## 6. Yêu cầu triển khai code

- Tạo lớp `DecisionTreeRegressor` kế thừa từ `BaseRegressor`.
- Tự cài đặt thuật toán CART bằng NumPy, không sử dụng `sklearn.tree`.
- Sử dụng tiêu chí giảm SSE (tương đương giảm MSE khi so sánh các cách chia trên cùng một node).
- Phải hỗ trợ các tham số `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` và `random_state`.
- Khi tìm điểm chia, chỉ xét các ngưỡng nằm giữa hai giá trị đặc trưng khác nhau và bảo đảm mỗi nhánh có đủ số mẫu tối thiểu.
- Có thể sử dụng `np.argsort`, `np.cumsum` để tăng hiệu quả tìm điểm chia.
- Mỗi node cần lưu thông tin đặc trưng, ngưỡng, node con trái, node con phải và giá trị dự đoán.
- Cần có phương thức `fit()` để xây dựng cây và `predict()` để dự đoán.
- Kiểm tra dữ liệu đầu vào và thông báo lỗi khi mô hình chưa được huấn luyện.
- Hỗ trợ lưu và khôi phục trạng thái cây thông qua `get_state()` và `_set_learned_state()` của lớp cơ sở.
