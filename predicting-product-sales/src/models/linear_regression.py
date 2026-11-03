"""LinearRegression tự cài: ridge, nghiệm đóng (không phạt intercept)."""
import numpy as np

from src.models.base import BaseRegressor


class LinearRegression(BaseRegressor):
    def __init__(self, alpha: float = 0.0):
        self.alpha = alpha  # hệ số phạt L2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("X phải là ma trận 2D và số dòng bằng số phần tử của y")

        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_[self.x_std_ == 0] = 1.0  # cột hằng số

        Xs = (X - self.x_mean_) / self.x_std_
        Xb = np.hstack([np.ones((X.shape[0], 1)), Xs])

        A = Xb.T @ Xb
        b = Xb.T @ y
        penalty = np.eye(Xb.shape[1]) * self.alpha
        penalty[0, 0] = 0.0  # không phạt intercept
        try:
            w = np.linalg.solve(A + penalty, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A + penalty, b, rcond=None)[0]

        self.w_ = w
        self.coef_ = w[1:] / self.x_std_          # theo thang đặc trưng gốc
        self.intercept_ = float(w[0] - self.coef_ @ self.x_mean_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "w_"):
            raise RuntimeError("Mô hình chưa được fit — gọi fit() trước predict()")
        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mean_) / self.x_std_
        Xb = np.hstack([np.ones((X.shape[0], 1)), Xs])
        return Xb @ self.w_

    def _learned_state(self) -> dict:
        return {"w_": self.w_, "x_mean_": self.x_mean_, "x_std_": self.x_std_}

    def _set_learned_state(self, learned: dict) -> None:
        self.w_ = np.asarray(learned["w_"], dtype=float)
        self.x_mean_ = np.asarray(learned["x_mean_"], dtype=float)
        self.x_std_ = np.asarray(learned["x_std_"], dtype=float)
        self.coef_ = self.w_[1:] / self.x_std_
        self.intercept_ = float(self.w_[0] - self.coef_ @ self.x_mean_)
