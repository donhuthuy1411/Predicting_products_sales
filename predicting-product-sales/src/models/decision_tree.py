"""DecisionTreeRegressor (CART) tự cài: tiêu chí tách giảm MSE, tìm điểm tách O(n log n)."""
import numpy as np

from src.models.base import BaseRegressor

# Cây lưu bằng structured array của numpy (lá có feature = -1 thay cho None)
NODE_DTYPE = np.dtype([
    ("feature", np.float64),
    ("threshold", np.float64),
    ("left", np.int64),
    ("right", np.int64),
    ("value", np.float64),
])
NODE_DTYPE_TREE = np.dtype(NODE_DTYPE.descr + [("tree", np.int64)])  # nhiều cây trong 1 mảng (cho forest/boosting)


def nodes_to_array(nodes: list[dict]) -> np.ndarray:
    return np.array(
        [(-1.0 if n["feature"] is None else float(n["feature"]), float(n["threshold"]),
          int(n["left"]), int(n["right"]), float(n["value"])) for n in nodes],
        dtype=NODE_DTYPE,
    )


def array_to_nodes(arr: np.ndarray) -> list[dict]:
    return [
        {"feature": None if r["feature"] < 0 else int(r["feature"]), "threshold": float(r["threshold"]),
         "left": int(r["left"]), "right": int(r["right"]), "value": float(r["value"])}
        for r in arr
    ]


class DecisionTreeRegressor(BaseRegressor):
    def __init__(
        self,
        max_depth: int = 8,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: int | float | str | None = None,
        random_state: int | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def _feature_subset(self, p: int, rng: np.random.Generator) -> np.ndarray:
        mf = self.max_features
        if mf is None:
            return np.arange(p)
        if isinstance(mf, str):
            if mf == "sqrt":
                k = max(1, int(np.sqrt(p)))
            elif mf == "third":
                k = max(1, p // 3)
            else:
                raise ValueError(f"max_features không hợp lệ: {mf!r}")
        elif isinstance(mf, float):
            if not 0 < mf <= 1:
                raise ValueError("max_features dạng float phải nằm trong (0, 1]")
            k = max(1, int(p * mf))
        else:
            k = int(mf)
        k = min(k, p)
        return rng.choice(p, size=k, replace=False)

    def _best_split(self, X: np.ndarray, y: np.ndarray, feat_idx: np.ndarray) -> tuple[float, int, float] | None:
        """Tìm (sse, feature, threshold) tốt nhất bằng tổng tích lũy — O(n log n) mỗi đặc trưng."""
        n = len(y)
        total_y = float(y.sum())
        total_y2 = float(np.sum(y * y))
        best = None
        for j in feat_idx:
            x = X[:, j]
            order = np.argsort(x, kind="mergesort")
            xs, ys = x[order], y[order]
            cy = np.cumsum(ys)
            cy2 = np.cumsum(ys * ys)
            i = np.arange(n - 1)
            # điểm tách hợp lệ: giữa hai giá trị khác nhau, đủ min_samples_leaf mỗi nhánh
            valid = (xs[:-1] < xs[1:]) & (i + 1 >= self.min_samples_leaf) & (n - (i + 1) >= self.min_samples_leaf)
            if not valid.any():
                continue
            left_n = i[valid] + 1
            left_y, left_y2 = cy[:-1][valid], cy2[:-1][valid]
            right_n = n - left_n
            right_y, right_y2 = total_y - left_y, total_y2 - left_y2
            sse = (left_y2 - left_y**2 / left_n) + (right_y2 - right_y**2 / right_n)
            k = int(np.argmin(sse))
            if best is None or sse[k] < best[0]:
                xl = xs[:-1][valid]  # giá trị bên trái điểm tách
                xr = xs[1:][valid]   # giá trị bên phải điểm tách
                thr = float((xl[k] + xr[k]) / 2.0)
                best = (float(sse[k]), int(j), thr)
        return best

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator) -> int:
        node_idx = len(self.nodes_)
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or np.all(y == y[0])
        ):
            self.nodes_.append({"feature": None, "threshold": 0.0, "left": -1, "right": -1, "value": float(y.mean())})
            return node_idx

        feat_idx = self._feature_subset(X.shape[1], rng)
        best = self._best_split(X, y, feat_idx)
        if best is None:
            self.nodes_.append({"feature": None, "threshold": 0.0, "left": -1, "right": -1, "value": float(y.mean())})
            return node_idx

        _, j, thr = best
        self.nodes_.append({"feature": j, "threshold": thr, "left": -1, "right": -1, "value": float(y.mean())})
        mask = X[:, j] <= thr
        left = self._build(X[mask], y[mask], depth + 1, rng)
        right = self._build(X[~mask], y[~mask], depth + 1, rng)
        self.nodes_[node_idx]["left"] = left
        self.nodes_[node_idx]["right"] = right
        return node_idx

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("X phải là ma trận 2D và số dòng bằng số phần tử của y")
        self.n_features_in_ = X.shape[1]
        self.nodes_ = []
        rng = np.random.default_rng(self.random_state)
        self._build(X, y, depth=0, rng=rng)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "nodes_"):
            raise RuntimeError("Mô hình chưa được fit — gọi fit() trước predict()")
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        cur = np.zeros(n, dtype=int)
        # các node lưu theo thứ tự tạo (cha trước con) nên duyệt một lượt là đủ
        for idx, node in enumerate(self.nodes_):
            if node["feature"] is None:
                continue
            m = cur == idx
            if not m.any():
                continue
            go_left = X[m, node["feature"]] <= node["threshold"]
            cur[m] = np.where(go_left, node["left"], node["right"])
        return np.array([self.nodes_[i]["value"] for i in cur])

    def _learned_state(self) -> dict:
        return {"nodes": nodes_to_array(self.nodes_), "n_features_in_": int(self.n_features_in_)}

    def _set_learned_state(self, learned: dict) -> None:
        self.nodes_ = array_to_nodes(np.asarray(learned["nodes"]))
        self.n_features_in_ = int(learned["n_features_in_"])
