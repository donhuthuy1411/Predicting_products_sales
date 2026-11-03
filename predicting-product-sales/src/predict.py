"""SalesPredictor: nạp preprocessor + mô hình tốt nhất để dự đoán dữ liệu mới đúng pipeline train."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CUSTOMER_ID_COL,
    DATE_COL,
    HOLIDAY_COL,
    MODELS_DIR,
    PRODUCT_ID_COL,
)
from src.features.customer_value import CustomerValueFeatures  # noqa: F401 (cần import lớp để unpickle preprocessor)
from src.features.popularity import PopularityFeatures  # noqa: F401
from src.features.seasonal import add_seasonal_features
from src.models.base import load_model


class SalesPredictor:
    def __init__(self, preprocessor: dict, model, feature_columns: list[str]) -> None:
        self.pre = preprocessor
        self.model = model
        self.feature_columns = feature_columns

    @classmethod
    def load(cls, models_dir: Path | None = None) -> "SalesPredictor":
        """Đọc preprocessor.npy, best_model.txt và <tên>.npy từ models_dir (mặc định MODELS_DIR)."""
        models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
        pre_path = models_dir / "preprocessor.npy"
        best_path = models_dir / "best_model.txt"
        if not pre_path.exists() or not best_path.exists():
            raise FileNotFoundError(
                f"Thiếu {pre_path} hoặc {best_path} — hãy chạy trước: "
                f"python -m scripts.prepare_data và python -m scripts.train"
            )
        pre = np.load(pre_path, allow_pickle=True).item()
        best = best_path.read_text(encoding="utf-8").strip()
        model_path = models_dir / f"{best}.npy"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Thiếu {model_path} (mô hình tốt nhất: {best}) — hãy chạy python -m scripts.train trước"
            )
        model = load_model(model_path)
        return cls(pre, model, pre["feature_columns"])

    def predict(self, df_raw: pd.DataFrame) -> np.ndarray:
        """Áp đúng thứ tự như prepare_data rồi dự đoán; kết quả kẹp >= 0. Không cần TARGET_COL."""
        df = df_raw.copy()
        df = self.pre["cleaner"].transform(df)
        df = add_seasonal_features(df, DATE_COL, HOLIDAY_COL)
        df = self.pre["popularity"].transform(df)
        df = self.pre["customer"].transform(df)
        drop_cols = [c for c in (DATE_COL, PRODUCT_ID_COL, CUSTOMER_ID_COL) if c is not None and c in df.columns]
        df = df.drop(columns=drop_cols)
        df = self.pre["encoder"].transform(df)
        df = df.reindex(columns=self.feature_columns, fill_value=0.0)
        X = df.to_numpy(dtype=float)
        return np.clip(self.model.predict(X), 0.0, None)

    def predict_one(self, record: dict) -> float:
        """Bọc dict thành DataFrame 1 dòng rồi dự đoán."""
        return float(self.predict(pd.DataFrame([record]))[0])
