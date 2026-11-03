"""Đặc trưng độ phổ biến sản phẩm: số review, rating, tần suất sản phẩm (không đọc target)."""
import numpy as np
import pandas as pd

from src.config import PRODUCT_ID_COL, RATING_COL, REVIEW_COL


class PopularityFeatures:
    """fit học bảng tần suất sản phẩm trên train; sản phẩm mới khi transform -> 0."""

    def __init__(self) -> None:
        self.product_freq_: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame) -> "PopularityFeatures":
        if PRODUCT_ID_COL is not None and PRODUCT_ID_COL in train_df.columns:
            self.product_freq_ = train_df[PRODUCT_ID_COL].value_counts()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if REVIEW_COL is not None and REVIEW_COL in df.columns:
            log_reviews = np.log1p(out[REVIEW_COL].clip(lower=0))
            out["log_reviews"] = log_reviews
            if RATING_COL is not None and RATING_COL in df.columns:
                out["rating_x_reviews"] = out[RATING_COL] * log_reviews
        if self.product_freq_ is not None and PRODUCT_ID_COL is not None and PRODUCT_ID_COL in df.columns:
            out["product_freq"] = out[PRODUCT_ID_COL].map(self.product_freq_).fillna(0).astype(float)
        return out
