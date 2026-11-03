"""Tự viết k-fold cross-validation và grid search (không dùng sklearn)."""
import itertools
import time

import numpy as np
import pandas as pd

from src.metrics import mae, mse, r2, rmse

METRICS = {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def k_fold_indices(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Trả list (train_idx, val_idx); xáo trộn theo seed, các fold không giao nhau."""
    if k < 2 or k > n:
        raise ValueError(f"k phải nằm trong [2, n], nhận được k={k}, n={n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1
    folds = []
    start = 0
    for fs in fold_sizes:
        val_idx = perm[start : start + fs]
        train_idx = np.setdiff1d(perm, val_idx)
        folds.append((train_idx, val_idx))
        start += fs
    return folds


def cross_val_score(model, X: np.ndarray, y: np.ndarray, k: int, metric: str = "rmse", seed: int = 42) -> float:
    """Trung bình lỗi qua k fold; mỗi fold clone model rồi fit trên phần train của fold."""
    if metric not in METRICS:
        raise ValueError(f"metric phải là một trong {sorted(METRICS)}, nhận được {metric!r}")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    fn = METRICS[metric]
    scores = []
    for train_idx, val_idx in k_fold_indices(len(X), k, seed):
        m = model.clone()
        m.fit(X[train_idx], y[train_idx])
        scores.append(fn(y[val_idx], m.predict(X[val_idx])))
    return float(np.mean(scores))


def grid_search(
    model,
    param_grid: dict[str, list],
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    metric: str = "rmse",
    seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """Duyệt tích Descartes các tổ hợp; trả (best_params, results_df) sắp xếp tăng dần theo lỗi."""
    keys = list(param_grid)
    combos = list(itertools.product(*(param_grid[key] for key in keys)))
    rows = []
    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        m = model.clone()
        m.set_params(**params)
        t0 = time.perf_counter()
        score = cross_val_score(m, X, y, k=k, metric=metric, seed=seed)
        dt = time.perf_counter() - t0
        rows.append({**params, "mean_error": score, "time_sec": dt})
        print(f"  {i}/{len(combos)} {params} -> {metric}={score:.4f} ({dt:.1f}s)")
    results = pd.DataFrame(rows).sort_values("mean_error").reset_index(drop=True)
    best_params = {key: results.iloc[0][key] for key in keys}
    return best_params, results
