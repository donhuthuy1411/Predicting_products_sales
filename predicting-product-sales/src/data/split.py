"""Chia train/test tự viết: theo thời gian (test nằm sau train) hoặc ngẫu nhiên."""
import numpy as np
import pandas as pd


def train_test_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    date_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia df thành (train, test).

    - date_col không None: sắp xếp theo thời gian, test = phần cuối theo tỉ lệ test_size.
    - date_col None: xáo trộn bằng np.random.default_rng(seed) rồi chia.
    - test_size ngoài (0, 1) -> ValueError. Train và test không giao nhau (assert).
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size phải nằm trong (0, 1), nhận được {test_size}")

    n = len(df)
    n_test = max(1, int(n * test_size))

    if date_col is not None:
        if date_col not in df.columns:
            raise ValueError(f"Không có cột ngày {date_col!r} trong DataFrame")
        ordered = np.argsort(df[date_col].to_numpy(), kind="stable")
        test_pos = ordered[-n_test:]
        train_pos = ordered[:-n_test]
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        test_pos = perm[:n_test]
        train_pos = perm[n_test:]

    assert len(np.intersect1d(train_pos, test_pos)) == 0, "train và test giao nhau"

    train = df.iloc[train_pos].reset_index(drop=True)
    test = df.iloc[test_pos].reset_index(drop=True)
    return train, test
