"""Đánh giá 4 mô hình trên tập test, chọn mô hình tốt nhất, xuất bảng + hình (không train lại)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, TARGET_COL
from src.metrics import evaluate_all
from src.models.base import load_model

MODEL_NAMES = ["linear_regression", "decision_tree", "random_forest", "gradient_boosting"]


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "test.csv")
    y = df[TARGET_COL].to_numpy(dtype=float)
    X = df.drop(columns=[TARGET_COL]).to_numpy(dtype=float)

    rows = []
    preds = {}
    for name in MODEL_NAMES:
        model = load_model(MODELS_DIR / f"{name}.npy")
        y_pred = model.predict(X)
        preds[name] = y_pred
        m = evaluate_all(y, y_pred)
        rows.append({"model": name, **m})
        print(f"{name:20s} mse={m['mse']:.2f} mae={m['mae']:.2f} rmse={m['rmse']:.2f} r2={m['r2']:.4f}")

    results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    results.to_csv(REPORTS_DIR / "results.csv", index=False)

    best = results.iloc[0]["model"]
    (MODELS_DIR / "best_model.txt").write_text(best, encoding="utf-8")
    print(f"\nBảng kết quả (sắp xếp theo rmse) -> {REPORTS_DIR}/results.csv")
    print(results.round(4).to_string(index=False))
    print(f"Mô hình tốt nhất: {best} -> {MODELS_DIR}/best_model.txt")

    # RMSE các mô hình
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(results["model"], results["rmse"], color="steelblue")
    ax.set_ylabel("RMSE")
    ax.set_title("So sánh RMSE trên tập test")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eval_rmse_bar.png", dpi=120)
    plt.close(fig)

    # predicted vs actual của mô hình tốt nhất + đường y=x
    y_best = preds[best]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, y_best, s=8, alpha=0.5)
    lim = [min(y.min(), y_best.min()), max(y.max(), y_best.max())]
    ax.plot(lim, lim, "r--", label="y = x")
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title(f"Predicted vs Actual — {best}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eval_pred_vs_actual.png", dpi=120)
    plt.close(fig)

    # histogram phần dư
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y - y_best, bins=50, color="gray", edgecolor="black")
    ax.set_title(f"Phần dư (actual - predicted) — {best}")
    ax.set_xlabel("residual")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eval_residual_hist.png", dpi=120)
    plt.close(fig)
    print(f"Đã lưu hình đánh giá vào {FIGURES_DIR}")


if __name__ == "__main__":
    main()
