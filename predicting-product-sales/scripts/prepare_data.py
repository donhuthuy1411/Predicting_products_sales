"""Pipeline tiền xử lý end-to-end: load -> clean -> split -> features -> encode -> lưu."""
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_COLS,
    CUSTOMER_ID_COL,
    DATE_COL,
    HOLIDAY_COL,
    MODELS_DIR,
    PROCESSED_DIR,
    PRODUCT_ID_COL,
    SEED,
    TARGET_COL,
    TEST_SIZE,
)
from src.data.cleaning import Cleaner, drop_invalid_rows
from src.data.encoding import CategoricalEncoder
from src.data.loader import load_raw
from src.data.split import train_test_split
from src.features.customer_value import CustomerValueFeatures
from src.features.popularity import PopularityFeatures
from src.features.seasonal import add_seasonal_features


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1-2. load + bỏ dòng không hợp lệ
    df = load_raw()
    df = drop_invalid_rows(df)
    print(f"sau drop_invalid_rows: {df.shape}")

    # 3. chia train/test (không có cột ngày -> ngẫu nhiên theo seed)
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, seed=SEED, date_col=DATE_COL)
    print(f"train: {train_df.shape}, test: {test_df.shape}")

    # 4. Cleaner học trên train rồi áp cho cả hai
    cleaner = Cleaner().fit(train_df)
    train_df = cleaner.transform(train_df)
    test_df = cleaner.transform(test_df)

    # 5-7. đặc trưng mùa vụ, độ phổ biến, giá trị khách (fit trên train)
    train_df = add_seasonal_features(train_df, DATE_COL, HOLIDAY_COL)
    test_df = add_seasonal_features(test_df, DATE_COL, HOLIDAY_COL)

    popularity = PopularityFeatures().fit(train_df)
    train_df = popularity.transform(train_df)
    test_df = popularity.transform(test_df)

    customer = CustomerValueFeatures().fit(train_df)
    train_df = customer.transform(train_df)
    test_df = customer.transform(test_df)

    # 8. bỏ cột ngày/ID (không đưa vào mô hình)
    drop_cols = [c for c in (DATE_COL, PRODUCT_ID_COL, CUSTOMER_ID_COL) if c is not None and c in train_df.columns]
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    # 9. one-hot (fit trên train)
    encoder = CategoricalEncoder().fit(train_df, CATEGORICAL_COLS)
    train_df = encoder.transform(train_df)
    test_df = encoder.transform(test_df)

    # 10. căn cột test theo train (thiếu -> 0), target đặt ở cột cuối, chỉ còn cột số
    feature_columns = [c for c in train_df.columns if c != TARGET_COL]
    test_df = test_df.reindex(columns=feature_columns + [TARGET_COL], fill_value=0.0)
    train_df = train_df[feature_columns + [TARGET_COL]]

    non_numeric = [c for c in train_df.columns if not pd.api.types.is_numeric_dtype(train_df[c])]
    if non_numeric:
        raise ValueError(f"Vẫn còn cột không phải số sau encode: {non_numeric}")
    if train_df.isna().any().any() or test_df.isna().any().any():
        raise ValueError("Vẫn còn giá trị thiếu sau tiền xử lý")

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    preprocessor = {
        "cleaner": cleaner,
        "popularity": popularity,
        "customer": customer,
        "encoder": encoder,
        "feature_columns": feature_columns,
    }
    np.save(MODELS_DIR / "preprocessor.npy", preprocessor, allow_pickle=True)

    print(f"\nĐã lưu {PROCESSED_DIR}/train.csv ({train_df.shape}) và test.csv ({test_df.shape})")
    print(f"Đã lưu {MODELS_DIR}/preprocessor.npy")
    print(f"số đặc trưng: {len(feature_columns)}")
    print("danh sách đặc trưng:")
    for c in feature_columns:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
