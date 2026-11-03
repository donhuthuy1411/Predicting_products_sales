# K-Fold Cross-Validation và Grid Search

> Mã nguồn: `src/tuning.py` | Sử dụng cho: Linear Regression, Decision Tree, Random Forest và Gradient Boosting

## 1. Mục tiêu

Xây dựng phương pháp đánh giá và lựa chọn siêu tham số phù hợp cho các mô hình hồi quy trong dự án.

K-Fold Cross-Validation chia dữ liệu huấn luyện thành nhiều phần nhỏ. Mỗi lần chạy, một phần được dùng để kiểm tra và những phần còn lại được dùng để huấn luyện. Quá trình được lặp lại cho đến khi tất cả các phần đều được dùng làm tập validation.

Grid Search kết hợp với K-Fold để thử nhiều tổ hợp siêu tham số và lựa chọn cấu hình có sai số trung bình thấp nhất.

## 2. Input / Output

|                            | Kiểu                | Shape    | Ý nghĩa                                            |
| -------------------------- | ------------------- | -------- | -------------------------------------------------- |
| Input `model`              | `BaseRegressor`     | —        | Mô hình cần tối ưu tham số                         |
| Input `X`                  | `np.ndarray`        | `(n, p)` | Ma trận đặc trưng                                  |
| Input `y`                  | `np.ndarray`        | `(n,)`   | Doanh số thực tế                                   |
| Input `param_grid`         | `dict`              | —        | Danh sách các giá trị tham số cần thử              |
| Output `k_fold_indices()`  | `list`              | `k` phần | Danh sách chỉ số train và validation của từng fold |
| Output `cross_val_score()` | `float`             | —        | Sai số trung bình qua các fold                     |
| Output `grid_search()`     | `dict`, `DataFrame` | —        | Bộ tham số tốt nhất và bảng kết quả thử nghiệm     |

Tham số chính:

- `k=5`: số lượng fold.
- `metric="rmse"`: chỉ số dùng để đánh giá.
- `seed=42`: kiểm soát quá trình xáo trộn dữ liệu.

Các metric được hỗ trợ gồm `mse`, `mae`, `rmse` và `r2`.

## 3. Công thức và tiêu chí đánh giá

Với K-Fold Cross-Validation, dữ liệu được chia thành `k` phần. Ở mỗi lần chạy, một phần được chọn làm validation và các phần còn lại làm train.

Điểm đánh giá trung bình được tính:

`CV Score = (1/k) * Σ Score_i`

Trong đó `Score_i` là kết quả đánh giá tại fold thứ i.

Grid Search thực hiện thử tất cả các tổ hợp tham số trong `param_grid`. Cấu hình có giá trị `mean_error` thấp nhất sẽ được chọn làm cấu hình tốt nhất.

Trong dự án, RMSE được sử dụng làm metric mặc định:

`RMSE = sqrt((1/n) * Σ(yi - y_pred_i)²)`

Kết quả tuning trên tập train (`reports/tuning_results.csv`):

| Mô hình           | Siêu tham số được chọn                                  | CV RMSE |
| ----------------- | ------------------------------------------------------- | ------: |
| Linear Regression | `alpha=10`                                              | 1138.49 |
| Decision Tree     | `max_depth=4`, `min_samples_leaf=20`                    | 1121.37 |
| Random Forest     | `n_estimators=60`, `max_depth=12`                       | 1095.56 |
| Gradient Boosting | `n_estimators=100`, `learning_rate=0.05`, `max_depth=3` | 1087.02 |

## 4. Cách hoạt động (How it works)

1. Nhận số lượng mẫu `n`, số fold `k` và seed để tạo danh sách chỉ số dữ liệu.
2. Xáo trộn chỉ số mẫu bằng bộ sinh số ngẫu nhiên có seed cố định.
3. Chia các chỉ số thành `k` phần có kích thước gần bằng nhau. Ở mỗi lần chạy, một phần được dùng làm validation, các phần còn lại dùng làm train.
4. Tạo một bản sao mới của mô hình bằng `clone()` để tránh sử dụng lại trạng thái đã huấn luyện.
5. Huấn luyện mô hình trên phần train của từng fold, sau đó dự đoán và tính metric trên phần validation.
6. Tính trung bình kết quả của tất cả các fold để thu được điểm đánh giá của một cấu hình tham số.
7. Grid Search sử dụng `itertools.product` để tạo tất cả tổ hợp tham số từ `param_grid`.
8. Lần lượt thử từng tổ hợp, thực hiện Cross-Validation và lưu lại tham số, sai số trung bình cùng thời gian chạy.
9. Sắp xếp kết quả theo `mean_error` tăng dần và chọn tổ hợp có sai số thấp nhất.

Sau khi tìm được tham số tốt nhất, mô hình được huấn luyện lại trên toàn bộ tập train trước khi lưu để sử dụng.

## 5. Dữ liệu sử dụng (Data used)

- Nguồn dữ liệu: `data/processed/train.csv`.
- Tập train gồm 6.819 mẫu và 52 đặc trưng.
- Biến mục tiêu là `Item_Outlet_Sales`.
- Dữ liệu train được chia thành 5 fold trong quá trình Cross-Validation.
- Tập test gồm 1.704 mẫu và không được sử dụng trong quá trình Grid Search.

Việc giữ riêng tập test giúp đánh giá khách quan khả năng dự đoán của mô hình sau khi đã hoàn tất quá trình lựa chọn siêu tham số.

## 6. Yêu cầu triển khai code

- Tự cài đặt K-Fold Cross-Validation và Grid Search bằng Python, NumPy và Pandas.
- Không sử dụng `sklearn.model_selection.KFold`, `cross_val_score` hoặc `GridSearchCV`.
- Xây dựng các hàm `k_fold_indices()`, `cross_val_score()` và `grid_search()`.
- `k_fold_indices()` phải kiểm tra điều kiện `2 <= k <= n`.
- Các fold không được trùng lặp chỉ số validation và phải sử dụng hết các mẫu.
- Mỗi fold phải tạo mô hình mới bằng `clone()` trước khi huấn luyện.
- Hỗ trợ các metric `mse`, `mae`, `rmse` và `r2`.
- Grid Search phải duyệt toàn bộ tổ hợp tham số được cung cấp.
- Kết quả cần lưu tham số, `mean_error` và `time_sec`.
- Sắp xếp kết quả theo sai số trung bình tăng dần để chọn cấu hình tốt nhất.
- Không sử dụng tập test trong quá trình tuning.
- Phải trả về `best_params` và bảng kết quả để các bước huấn luyện tiếp theo có thể sử dụng.
