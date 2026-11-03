"""GradientBoostingRegressor tự cài: loss bình phương, mỗi vòng học một cây nông trên residual."""
import numpy as np

from src.models.base import BaseRegressor
from src.models.decision_tree import (
    NODE_DTYPE_TREE,
    DecisionTreeRegressor,
    array_to_nodes,
    nodes_to_array,
)


class GradientBoostingRegressor(BaseRegressor):
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        subsample: float = 1.0,
        random_state: int | None = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        if not 0 < self.subsample <= 1.0:
            raise ValueError("subsample phải nằm trong (0, 1]")
        rng = np.random.default_rng(self.random_state)

        self.init_ = float(np.mean(y))
        F = np.full(n, self.init_)
        self.estimators_ = []
        self.train_loss_ = []
        for _ in range(self.n_estimators):
            residual = y - F
            if self.subsample < 1.0:
                k = max(1, int(self.subsample * n))
                idx = rng.choice(n, size=k, replace=False)  # mẫu con không hoàn lại
            else:
                idx = np.arange(n)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X[idx], residual[idx])
            F += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
            self.train_loss_.append(float(np.mean((y - F) ** 2)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("Mô hình chưa được fit — gọi fit() trước predict()")
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.init_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        return F

    def _learned_state(self) -> dict:
        rows = []
        for t, tree in enumerate(self.estimators_):
            for r in nodes_to_array(tree.nodes_):
                rows.append((r["feature"], r["threshold"], r["left"], r["right"], r["value"], t))
        nodes = np.array(rows, dtype=NODE_DTYPE_TREE) if rows else np.empty(0, dtype=NODE_DTYPE_TREE)
        tree_params = self.estimators_[0].get_params() if self.estimators_ else {}
        n_feat = int(self.estimators_[0].n_features_in_) if self.estimators_ else 0
        return {
            "nodes": nodes,
            "tree_params": tree_params,
            "n_features_in_": n_feat,
            "init_": float(self.init_),
            "train_loss_": np.asarray(self.train_loss_, dtype=float),
        }

    def _set_learned_state(self, learned: dict) -> None:
        nodes = np.asarray(learned["nodes"])
        tree_params = dict(learned["tree_params"])
        n_feat = int(learned["n_features_in_"])
        self.estimators_ = []
        if len(nodes):
            for t in range(int(nodes["tree"].max()) + 1):
                tree = DecisionTreeRegressor(**tree_params)
                tree.nodes_ = array_to_nodes(nodes[nodes["tree"] == t])
                tree.n_features_in_ = n_feat
                self.estimators_.append(tree)
        self.init_ = float(learned["init_"])
        self.train_loss_ = [float(v) for v in np.asarray(learned["train_loss_"], dtype=float)]
