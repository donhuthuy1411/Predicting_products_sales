"""Đặc trưng giá trị khách hàng (CLV proxy, tần suất mua) tính trên train; khách mới -> 0."""
import pandas as pd

from src.config import CUSTOMER_ID_COL, DATE_COL, PRICE_COL, QUANTITY_COL


class CustomerValueFeatures:
    """fit học bảng theo khách trên train; transform tra bảng, khách mới/không có mã -> 0."""

    def __init__(self) -> None:
        self.table_: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> "CustomerValueFeatures":
        if CUSTOMER_ID_COL is None or CUSTOMER_ID_COL not in train_df.columns:
            return self

        g = train_df.groupby(CUSTOMER_ID_COL, observed=True)
        table = pd.DataFrame({"n_purchases": g.size()})

        if PRICE_COL is not None and PRICE_COL in train_df.columns:
            if QUANTITY_COL is not None and QUANTITY_COL in train_df.columns:
                clv = train_df[PRICE_COL] * train_df[QUANTITY_COL]
            else:
                clv = train_df[PRICE_COL]
            table["clv_proxy"] = clv.groupby(train_df[CUSTOMER_ID_COL], observed=True).sum()
        else:
            table["clv_proxy"] = 0.0

        if DATE_COL is not None and DATE_COL in train_df.columns:
            span = g[DATE_COL].agg(lambda s: (s.max() - s.min()).days / 30.0)
            months_active = span.clip(lower=1.0)  # tối thiểu 1 tháng
            table["purchase_freq"] = table["n_purchases"] / months_active
        else:
            table["purchase_freq"] = table["n_purchases"].astype(float)

        self.table_ = table
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.table_ is None or CUSTOMER_ID_COL is None or CUSTOMER_ID_COL not in df.columns:
            return out
        mapped = out[CUSTOMER_ID_COL].map(self.table_)
        for col in ("n_purchases", "clv_proxy", "purchase_freq"):
            out[col] = mapped[col].fillna(0.0)
        return out
