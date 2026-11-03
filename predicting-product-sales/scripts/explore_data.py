"""EDA: in thống kê ra console và lưu hình vào reports/figures/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CATEGORICAL_COLS, DATE_COL, FIGURES_DIR, NUMERIC_COLS, TARGET_COL
from src.data.loader import load_raw


def count_iqr_outliers(s: pd.Series, k: float = 1.5) -> int:
    """Số ngoại lai theo quy tắc IQR với hệ số k."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return int(((s < lower) | (s > upper)).sum())


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()

    print("=" * 60)
    print("EDA — Predicting Product Sales")
    print("=" * 60)
    print(f"shape: {df.shape[0]} dòng x {df.shape[1]} cột")
    print(f"số dòng trùng hoàn toàn: {int(df.duplicated().sum())}")

    print("\n-- dtypes --")
    print(df.dtypes.to_string())

    print("\n-- % giá trị thiếu theo cột --")
    pct_missing = df.isna().mean() * 100
    print(pct_missing.round(2).to_string())

    print("\n-- describe() các cột số --")
    print(df[NUMERIC_COLS + [TARGET_COL]].describe().round(3).to_string())

    print("\n-- số giá trị duy nhất của cột phân loại --")
    for c in CATEGORICAL_COLS:
        print(f"  {c}: {df[c].nunique(dropna=True)} (thiếu: {int(df[c].isna().sum())})")

    print("\n-- số ngoại lai theo IQR (k=1.5) --")
    for c in [TARGET_COL, *NUMERIC_COLS]:
        print(f"  {c}: {count_iqr_outliers(df[c].dropna())}")

    # Histogram của target
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[TARGET_COL].dropna(), bins=50, color="steelblue", edgecolor="black")
    ax.set_title(f"Phân phối {TARGET_COL}")
    ax.set_xlabel(TARGET_COL)
    ax.set_ylabel("số dòng")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_target_hist.png", dpi=120)
    plt.close(fig)

    # Ma trận tương quan các cột số
    corr = df[NUMERIC_COLS + [TARGET_COL]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Tương quan các cột số")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_corr_heatmap.png", dpi=120)
    plt.close(fig)

    # Trung bình target theo top 10 giá trị của mỗi cột phân loại
    for c in CATEGORICAL_COLS:
        top = df[c].value_counts().head(10).index
        sub = df[df[c].isin(top)].groupby(c, observed=True)[TARGET_COL].mean()
        sub = sub.sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(sub)), sub.values, color="seagreen")
        ax.set_xticks(range(len(sub)), [str(v) for v in sub.index], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Trung bình {TARGET_COL} theo {c} (top 10)")
        ax.set_ylabel(f"mean {TARGET_COL}")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"eda_target_by_{c}.png", dpi=120)
        plt.close(fig)

    # Tổng doanh số theo tháng nếu có cột ngày
    if DATE_COL is not None:
        monthly = df.set_index(DATE_COL)[TARGET_COL].resample("ME").sum()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(monthly.index, monthly.values, marker="o", color="darkorange")
        ax.set_title(f"Tổng {TARGET_COL} theo tháng")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "eda_sales_over_time.png", dpi=120)
        plt.close(fig)

    print(f"\nĐã lưu hình EDA vào {FIGURES_DIR}")


if __name__ == "__main__":
    main()
