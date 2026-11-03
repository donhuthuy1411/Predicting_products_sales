"""Đặc trưng mùa vụ/lễ từ cột ngày (stateless — không học gì từ dữ liệu)."""
import numpy as np
import pandas as pd


def add_seasonal_features(
    df: pd.DataFrame,
    date_col: str | None,
    holiday_col: str | None = None,
) -> pd.DataFrame:
    """Thêm month/quý/thứ/cuối tuần/lễ + sin-cos chu kỳ. Trả về bản sao, không xóa cột ngày gốc.

    date_col None -> trả về bản sao nguyên vẹn. Giá trị ngày NaT -> các cột mới = 0.
    """
    out = df.copy()
    if date_col is None:
        return out
    if date_col not in out.columns:
        return out

    dt = pd.to_datetime(out[date_col], errors="coerce")
    valid = dt.notna()

    month = dt.dt.month.fillna(0).astype(int)
    dow = dt.dt.dayofweek.fillna(0).astype(int)

    out["month"] = month
    out["quarter"] = dt.dt.quarter.fillna(0).astype(int)
    out["day_of_week"] = dow
    out["is_weekend"] = (dow >= 5).astype(int)
    out["month_sin"] = np.where(valid, np.sin(2 * np.pi * (month - 1) / 12), 0.0)
    out["month_cos"] = np.where(valid, np.cos(2 * np.pi * (month - 1) / 12), 0.0)
    out["dow_sin"] = np.where(valid, np.sin(2 * np.pi * dow / 7), 0.0)
    out["dow_cos"] = np.where(valid, np.cos(2 * np.pi * dow / 7), 0.0)

    if holiday_col is not None and holiday_col in out.columns:
        out["is_holiday"] = (pd.to_numeric(out[holiday_col], errors="coerce").fillna(0) > 0).astype(int)

    return out
