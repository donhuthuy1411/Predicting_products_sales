"""CLI dự đoán doanh số theo kịch bản: --input scenarios.json hoặc --demo."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AD_SPEND_COL, DATE_COL, HOLIDAY_COL, PRODUCT_ID_COL, RATING_COL, REVIEW_COL
from src.data.loader import load_raw
from src.predict import SalesPredictor


def baseline_row(df_raw: pd.DataFrame) -> dict:
    """Một dòng 'điển hình' từ data/raw: median cho cột số, mode cho cột phân loại, ngày mới nhất."""
    row = {}
    for c in df_raw.columns:
        s = df_raw[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            row[c] = float(s.median())
        elif pd.api.types.is_datetime64_any_dtype(s):
            row[c] = s.max()
        else:
            mode = s.mode()
            row[c] = mode.iloc[0] if len(mode) > 0 else s.iloc[0]
    return row


def build_demo_scenarios(df_raw: pd.DataFrame, baseline: dict) -> list[tuple[str, dict]]:
    scenarios = [("baseline (điển hình)", dict(baseline))]

    # (a) sản phẩm mới ra mắt: review = 0, rating = trung bình;
    # nếu dataset không có review/rating thì dùng mã sản phẩm mới (product_freq = 0)
    if REVIEW_COL is not None and RATING_COL is not None:
        sc = dict(baseline)
        sc[REVIEW_COL] = 0
        sc[RATING_COL] = float(df_raw[RATING_COL].mean())
        scenarios.append(("sản phẩm mới (review=0, rating=mean)", sc))
    elif PRODUCT_ID_COL is not None:
        sc = dict(baseline)
        sc[PRODUCT_ID_COL] = "__SAN_PHAM_MOI__"  # chưa từng thấy trong train -> product_freq = 0
        scenarios.append(("sản phẩm mới (product_freq=0)", sc))
    else:
        print(f"[bỏ qua] kịch bản sản phẩm mới: dataset không có REVIEW_COL/RATING_COL/PRODUCT_ID_COL")

    # (b) tăng chi phí quảng cáo thêm 50%
    if AD_SPEND_COL is not None:
        sc = dict(baseline)
        sc[AD_SPEND_COL] = float(baseline.get(AD_SPEND_COL, 0.0)) * 1.5
        scenarios.append(("tăng chi phí quảng cáo +50%", sc))
    else:
        print(f"[bỏ qua] kịch bản quảng cáo: dataset không có AD_SPEND_COL")

    # (c) chuyển sang ngày lễ / cuối năm
    if DATE_COL is not None:
        sc = dict(baseline)
        year = pd.to_datetime(df_raw[DATE_COL], errors="coerce").max().year
        sc[DATE_COL] = pd.Timestamp(year=year, month=12, day=25)
        if HOLIDAY_COL is not None:
            sc[HOLIDAY_COL] = 1
        scenarios.append((f"ngày lễ cuối năm ({year}-12-25)", sc))
    else:
        print(f"[bỏ qua] kịch bản ngày lễ/cuối năm: dataset không có DATE_COL")

    return scenarios


def print_table(predictor: SalesPredictor, scenarios: list[tuple[str, dict]]) -> None:
    df = pd.DataFrame([sc for _, sc in scenarios])
    preds = predictor.predict(df)
    base_pred = preds[0]
    print(f"\n{'kịch bản':45s} {'doanh số dự đoán':>18s} {'chênh lệch':>14s} {'%':>8s}")
    print("-" * 90)
    for (name, _), p in zip(scenarios, preds):
        delta = p - base_pred
        pct = (delta / base_pred * 100) if base_pred != 0 else float("nan")
        print(f"{name:45s} {p:18.2f} {delta:+14.2f} {pct:+7.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dự đoán doanh số theo kịch bản")
    parser.add_argument("--input", type=str, default=None, help="đường dẫn file scenarios.json (danh sách dict)")
    parser.add_argument("--demo", action="store_true", help="chạy 3 kịch bản demo từ dữ liệu thô")
    args = parser.parse_args()

    if not args.demo and args.input is None:
        parser.error("cần --demo hoặc --input scenarios.json")

    predictor = SalesPredictor.load()

    if args.demo:
        df_raw = load_raw()
        baseline = baseline_row(df_raw)
        scenarios = build_demo_scenarios(df_raw, baseline)
    else:
        with open(Path(args.input), encoding="utf-8") as f:
            records = json.load(f)
        scenarios = [(f"scenario {i + 1}", rec) for i, rec in enumerate(records)]

    print_table(predictor, scenarios)


if __name__ == "__main__":
    main()
