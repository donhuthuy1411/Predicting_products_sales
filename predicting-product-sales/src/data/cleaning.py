"""Làm sạch dữ liệu: drop_invalid_rows (stateless) và Cleaner (học thống kê trên train)."""
import numpy as np
import pandas as pd

from src.config import CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Bỏ dòng thiếu TARGET_COL, bỏ dòng trùng hoàn toàn, bỏ dòng TARGET_COL < 0. Stateless."""
    out = df.dropna(subset=[TARGET_COL])
    out = out.drop_duplicates()
    out = out[out[TARGET_COL] >= 0]
    return out.reset_index(drop=True)


class Cleaner:
    """Học median/mode/cận IQR trên train rồi áp để điền thiếu và kẹp ngoại lai.

    Không bao giờ đọc hay kẹp TARGET_COL; transform chạy được khi df không có target.
    """

    def __init__(self) -> None:
        self.numeric_medians_: dict[str, float] = {}
        self.categorical_modes_: dict[str, object] = {}
        self.iqr_bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> "Cleaner":
        for c in NUMERIC_COLS:
            self.numeric_medians_[c] = float(df[c].median())
            q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            iqr = q3 - q1
            self.iqr_bounds_[c] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
        for c in CATEGORICAL_COLS:
            mode = df[c].mode()
            self.categorical_modes_[c] = mode.iloc[0] if len(mode) > 0 else None
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in [*NUMERIC_COLS, *CATEGORICAL_COLS] if c not in df.columns]
        if missing:
            raise ValueError(f"Thiếu cột cần thiết khi transform: {missing}")

        out = df.copy()
        for c in NUMERIC_COLS:
            out[c] = out[c].fillna(self.numeric_medians_[c])
            lower, upper = self.iqr_bounds_[c]
            if c != TARGET_COL:  # không kẹp target
                out[c] = out[c].clip(lower=lower, upper=upper)
        for c in CATEGORICAL_COLS:
            if self.categorical_modes_[c] is not None:
                out[c] = out[c].fillna(self.categorical_modes_[c])
        return out
