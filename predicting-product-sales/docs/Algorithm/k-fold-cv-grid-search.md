# K-Fold Cross-Validation + Grid Search

> Mã nguồn: `src/tuning.py` | Sử dụng cho: bốn mô hình hồi quy | Kết quả: `reports/tuning_results.csv`

## 1. Mục tiêu

Kết hợp K-Fold Cross-Validation và Grid Search để lựa chọn bộ siêu tham số phù hợp cho từng mô hình. K-Fold đánh giá cấu hình qua nhiều lần chia train/validation; Grid Search duyệt các tổ hợp tham số. Tập test độc lập không được sử dụng trong quá trình tuning.

## 2. Input / Output

|                               | Kiểu                        | Shape / cấu trúc | Ý nghĩa                                            |
| ----------------------------- | --------------------------- | ---------------- | -------------------------------------------------- |
| Input `model`                 | `BaseRegressor`             | —                | Mô hình chưa được fit                              |
| Input `X`, `y`                | `np.ndarray`, `np.ndarray`  | `(n, p)`, `(n,)` | Dữ liệu huấn luyện và target                       |
| Input `param_grid`            | `dict[str, list]`           | —                | Các giá trị siêu tham số cần thử                   |
| Output `cross_val_score(...)` | `float`                     | —                | Lỗi trung bình trên k fold                         |
| Output `grid_search(...)`     | `best_params`, `results_df` | dict, DataFrame  | Bộ tham số tốt nhất và bảng kết quả của các tổ hợp |

Tham số mặc định: `k=5`, `metric="rmse"`, `seed=42`. Ví dụ `param_grid = {"max_depth": [4, 8, 12], "min_samples_leaf": [5, 20]}`.

## 3. Tiêu chí đánh giá (evaluation metric / criteria)

- Hỗ trợ các metric `mse`, `mae`, `rmse`, `r2` từ `src/metrics.py`.
- Metric mặc định là **RMSE**; với các metric lỗi, giá trị trung bình càng nhỏ càng tốt.
- Mỗi tổ hợp tham số được đánh giá trên đủ `k` fold; kết quả được tổng hợp bằng `mean_error`.
- Tổ hợp có `mean_error` thấp nhất được chọn làm `best_params`.

Kết quả tuning 5-fold trên tập train (`reports/tuning_results.csv`):

| Mô hình           | Siêu tham số được chọn                                  | CV RMSE |
| ----------------- | ------------------------------------------------------- | ------: |
| Linear Regression | `alpha=10`                                              | 1138.49 |
| Decision Tree     | `max_depth=4`, `min_samples_leaf=20`                    | 1121.37 |
| Random Forest     | `n_estimators=60`, `max_depth=12`                       | 1095.56 |
| Gradient Boosting | `n_estimators=100`, `learning_rate=0.05`, `max_depth=3` | 1087.02 |

## 4. Cách hoạt động (How it works)

1. **Chia fold**: `k_fold_indices(n, k, seed)` xáo trộn chỉ số bằng `np.random.default_rng(seed).permutation(n)`, sau đó chia thành `k` phần có kích thước chênh lệch nhiều nhất một mẫu.
2. **Cross-validation**: lần lượt chọn một fold làm validation và các fold còn lại làm train. Mỗi lượt tạo mô hình mới bằng `model.clone()`, fit trên phần train rồi tính metric trên phần validation.
3. **Tính điểm trung bình**: tổng hợp `k` giá trị metric thành điểm trung bình của cấu hình đang xét.
4. **Grid Search**: dùng `itertools.product` duyệt tất cả tổ hợp trong `param_grid`; với mỗi tổ hợp, tạo bản sao mô hình, thiết lập tham số và gọi cross-validation.
5. **Chọn cấu hình tốt nhất**: lưu kết quả các tổ hợp vào DataFrame, sắp xếp theo `mean_error` tăng dần và lấy dòng đầu làm `best_params`.
6. **Huấn luyện mô hình cuối**: trong `scripts/train.py`, mô hình được fit lại trên toàn bộ tập train với tham số tốt nhất rồi mới lưu.

Tùy chọn `--fast` chỉ sử dụng 20% tập train cho bước tuning để rút ngắn thời gian chạy; mô hình cuối vẫn được fit trên toàn bộ tập train.

## 5. Dữ liệu sử dụng (Data used)

- Chỉ sử dụng `data/processed/train.csv` gồm 6.819 mẫu, 52 đặc trưng và target `Item_Outlet_Sales`.
- Dữ liệu train được chia thành 5 fold; mỗi fold lần lượt đóng vai trò validation một lần.
- `data/processed/test.csv` không được sử dụng trong Grid Search hoặc Cross-Validation; tập test chỉ được dùng ở bước đánh giá cuối cùng.
- Seed mặc định `42` giúp việc xáo trộn và chia fold có thể tái lập.
