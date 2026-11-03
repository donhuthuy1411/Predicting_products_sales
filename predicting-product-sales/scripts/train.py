"""Tune (k-fold grid search) + huấn luyện 4 mô hình trên tập train đã xử lý, lưu .npy (numpy)."""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src.config import K_FOLDS, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, SEED, TARGET_COL
from src.models.base import save_model
from src.models.decision_tree import DecisionTreeRegressor
from src.models.gradient_boosting import GradientBoostingRegressor
from src.models.linear_regression import LinearRegression
from src.models.random_forest import RandomForestRegressor
from src.tuning import grid_search

PARAM_GRIDS = {
    "linear_regression": (LinearRegression, {"alpha": [0, 1, 10, 100]}),
    "decision_tree": (
        DecisionTreeRegressor,
        {"max_depth": [4, 8, 12], "min_samples_leaf": [5, 20]},
    ),
    "random_forest": (
        RandomForestRegressor,
        {"n_estimators": [30, 60], "max_depth": [8, 12]},
    ),
    "gradient_boosting": (
        GradientBoostingRegressor,
        {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3]},
    ),
}


def _cast(v, grid_values):
    """Ép giá trị đọc từ DataFrame về đúng kiểu trong lưới gốc (int/float/bool) để ghi JSON và truyền lại model."""
    v = v.item() if isinstance(v, np.generic) else v
    ref = grid_values[0]
    if isinstance(ref, bool):
        return bool(v)
    if isinstance(ref, int):
        return int(v)
    if isinstance(ref, float):
        return float(v)
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune + train 4 mô hình")
    parser.add_argument("--fast", action="store_true",
                        help="chỉ dùng 20% train cho bước tuning rồi train lại bản tốt nhất trên toàn bộ train")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "train.csv")
    y = df[TARGET_COL].to_numpy(dtype=float)
    X = df.drop(columns=[TARGET_COL]).to_numpy(dtype=float)
    print(f"train: X={X.shape}, y={y.shape}" + ("  [chế độ --fast: tuning trên 20% train]" if args.fast else ""))

    if args.fast:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X), size=max(1, int(0.2 * len(X))), replace=False)
        X_tune, y_tune = X[idx], y[idx]
    else:
        X_tune, y_tune = X, y

    all_results = []
    best_params_all = {}
    for name, (cls, grid) in PARAM_GRIDS.items():
        print(f"\n=== {name} ===")
        base = cls(random_state=SEED) if cls is not LinearRegression else cls()
        t0 = time.perf_counter()
        best_params, results = grid_search(base, grid, X_tune, y_tune, k=K_FOLDS, metric="rmse", seed=SEED)
        results["model"] = name
        all_results.append(results)
        best_params = {k: _cast(v, grid[k]) for k, v in best_params.items()}
        best_params_all[name] = best_params

        final = cls(**best_params)
        if cls is not LinearRegression:
            final.set_params(random_state=SEED)
        t1 = time.perf_counter()
        final.fit(X, y)
        train_time = time.perf_counter() - t1

        save_model(final, MODELS_DIR / f"{name}.npy")
        print(f"best_params={best_params} | tune {t1 - t0:.1f}s, train full {train_time:.1f}s -> models/{name}.npy")

    tuning_df = pd.concat(all_results, ignore_index=True)
    tuning_df.to_csv(REPORTS_DIR / "tuning_results.csv", index=False)
    with open(MODELS_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_all, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {REPORTS_DIR}/tuning_results.csv và {MODELS_DIR}/best_params.json")


if __name__ == "__main__":
    main()
