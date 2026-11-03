"""Các chỉ số đánh giá hồi quy tự viết bằng numpy: MSE, MAE, RMSE, R2."""
import numpy as np


def _as_1d_float(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    if t.shape != p.shape:
        raise ValueError(f"Shape khác nhau: y_true {t.shape} vs y_pred {p.shape}")
    if t.size == 0:
        raise ValueError("Không được rỗng")
    return t, p


def mse(y_true, y_pred) -> float:
    t, p = _as_1d_float(y_true, y_pred)
    return float(np.mean((t - p) ** 2))


def mae(y_true, y_pred) -> float:
    t, p = _as_1d_float(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true, y_pred) -> float:
    t, p = _as_1d_float(y_true, y_pred)
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    ss_res = float(np.sum((t - p) ** 2))
    return 1.0 - ss_res / ss_tot


def evaluate_all(y_true, y_pred) -> dict[str, float]:
    return {"mse": mse(y_true, y_pred), "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred), "r2": r2(y_true, y_pred)}
