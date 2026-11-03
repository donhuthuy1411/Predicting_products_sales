# Decision Tree Regressor (CART)

> Mã nguồn: `src/models/decision_tree.py` | Thành phần dùng lại: Random Forest và Gradient Boosting | Đánh giá: `reports/results.csv`

## 1. Mục tiêu

Xây dựng cây hồi quy CART để chia không gian đặc trưng thành các vùng. Mỗi nút lá dự đoán bằng giá trị trung bình của target trong vùng đó. Đây là mô hình phi tuyến cơ bản và cũng là weak learner được sử dụng trong các mô hình ensemble.

## 2. Input / Output

|                          | Kiểu                       | Shape            | Ý nghĩa                                                     |
| ------------------------ | -------------------------- | ---------------- | ----------------------------------------------------------- |
| Input `fit(X, y)`        | `np.ndarray`, `np.ndarray` | `(n, p)`, `(n,)` | Ma trận đặc trưng và vector doanh số thực tế                |
| Output `predict(X)`      | `np.ndarray`               | `(n,)`           | Doanh số dự đoán cho từng mẫu                               |
| Model attribute `nodes_` | `list[dict]`               | —                | Danh sách node gồm feature, threshold, left, right và value |

Tham số chính: `max_depth=8`, `min_samples_split=10`, `min_samples_leaf=5`, `max_features=None`, `random_state=None`. `max_features` có thể là `None`, `"sqrt"`, `"third"`, số nguyên hoặc tỷ lệ trong `(0, 1]`. Gọi `predict` trước `fit` sẽ phát sinh `RuntimeError`.

## 3. Tiêu chí đánh giá (evaluation metric / criteria)

- **MAE**: sai số tuyệt đối trung bình.
- **MSE**: trung bình bình phương sai số.
- **RMSE**: căn bậc hai của MSE; được dùng làm metric chính khi tuning và so sánh.
- **R²**: hệ số xác định.
- Tuning dùng 5-fold Cross Validation với lưới `max_depth ∈ {4, 8, 12}` và `min_samples_leaf ∈ {5, 20}`.

Kết quả trên tập test (`reports/results.csv`):

|          MSE |    MAE |     RMSE |     R² |
| -----------: | -----: | -------: | -----: |
| 1,152,400.45 | 752.64 | 1,073.50 | 0.5989 |

Siêu tham số được chọn: `max_depth=4`, `min_samples_leaf=20` (CV RMSE = 1121.37).

## 4. Cách hoạt động (How it works)

1. **Tìm tiêu chí chia**: xét tổng bình phương sai lệch của hai nhánh `SSE_left + SSE_right` và chọn điểm chia có tổng nhỏ nhất.
2. **Tìm điểm chia**: với mỗi đặc trưng, sắp xếp giá trị một lần; dùng tổng tích lũy của `y` và `y²` (`np.cumsum`) để tính SSE tại các vị trí chia. Ngưỡng là trung điểm của hai giá trị khác nhau liền kề; chỉ xét vị trí bảo đảm mỗi nhánh có ít nhất `min_samples_leaf` mẫu.
3. **Chọn đặc trưng**: tại mỗi node, chọn tập đặc trưng cần xét theo `max_features`. `None` dùng tất cả; `"sqrt"` dùng căn bậc hai số đặc trưng; `"third"` dùng khoảng một phần ba; số nguyên hoặc tỷ lệ dùng số lượng tương ứng.
4. **Dừng phát triển cây**: dừng khi đạt `max_depth`, số mẫu nhỏ hơn `min_samples_split`, không tìm được phép chia hợp lệ hoặc target không thay đổi. Node lá lưu `value = mean(y)`.
5. **Dự đoán**: đưa từng mẫu đi qua các node theo điều kiện `feature_value <= threshold`; khi đến node lá, trả về `value` của node đó.

## 5. Dữ liệu sử dụng (Data used)

- Nguồn: `data/processed/train.csv` để huấn luyện và `data/processed/test.csv` để đánh giá.
- Tập train gồm 6.819 mẫu; tập test gồm 1.704 mẫu. Mỗi mẫu có 52 đặc trưng sau tiền xử lý; target là `Item_Outlet_Sales`.
- Mô hình sử dụng các đặc trưng số, đặc trưng `product_freq` và các cột One-Hot Encoding.
- Không đưa trực tiếp cột định danh đã loại bỏ trong bước tiền xử lý vào mô hình.
