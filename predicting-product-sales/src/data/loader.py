"""Đọc CSV gốc và kiểm tra tính hợp lệ cơ bản (không sửa dữ liệu)."""
from pathlib import Path

import pandas as pd

from src.config import CATEGORICAL_COLS, DATE_COL, NUMERIC_COLS, RAW_PATH, TARGET_COL


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Đọc CSV thô từ `path` (mặc định RAW_PATH trong config).

    - File không tồn tại -> FileNotFoundError nói rõ cần đặt file ở đâu.
    - Thiếu cột bắt buộc -> ValueError liệt kê cột thiếu.
    - Nếu có DATE_COL thì parse thành datetime (errors="coerce").
    - Không sửa giá trị, không xóa dòng.
    """
    path = Path(path) if path is not None else RAW_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {path}\n"
            f"Hãy tải dataset (Kaggle: BigMart Sales, file gốc tên Train.csv) đổi tên thành "
            f"bigmart_sales.csv rồi đặt vào: "
            f"{RAW_PATH}"
        )

    df = pd.read_csv(path)

    required = [TARGET_COL, *CATEGORICAL_COLS, *NUMERIC_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset thiếu các cột bắt buộc: {missing}. "
            f"Cột có trong file: {list(df.columns)}. "
            f"Nếu dùng dataset khác, hãy khai báo lại tên cột trong src/config.py."
        )

    if DATE_COL is not None and DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    return df
