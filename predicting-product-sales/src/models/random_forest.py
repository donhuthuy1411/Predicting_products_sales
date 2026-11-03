"""RandomForestRegressor tự cài: bagging trên các DecisionTreeRegressor, mỗi cây một bootstrap."""
import numpy as np

from src.models.base import BaseRegressor
from src.models.decision_tree import (
    NODE_DTYPE_TREE,
    DecisionTreeRegressor,
    array_to_nodes,
    nodes_to_array,
)


class RandomForestRegressor(BaseRegressor):
    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        max_features: int | float | str | None = "third",
        random_state: int | None = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            seed_i = int(rng.integers(0, 2**31 - 1))
            boot = rng.integers(0, n, size=n)  # bootstrap: lấy mẫu có hoàn lại
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed_i,
            )
            tree.fit(X[boot], y[boot])
            self.estimators_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("Mô hình chưa được fit — gọi fit() trước predict()")
        X = np.asarray(X, dtype=float)
        preds = np.stack([tree.predict(X) for tree in self.estimators_])
        return preds.mean(axis=0)

    def _learned_state(self) -> dict:
        rows = []
        for t, tree in enumerate(self.estimators_):
            for r in nodes_to_array(tree.nodes_):
                rows.append((r["feature"], r["threshold"], r["left"], r["right"], r["value"], t))
        nodes = np.array(rows, dtype=NODE_DTYPE_TREE) if rows else np.empty(0, dtype=NODE_DTYPE_TREE)
        tree_params = self.estimators_[0].get_params() if self.estimators_ else {}
        n_feat = int(self.estimators_[0].n_features_in_) if self.estimators_ else 0
        return {"nodes": nodes, "tree_params": tree_params, "n_features_in_": n_feat}

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
