"""CategoricalEncoder: one-hot tự viết, học danh mục trên train, pickle được."""
import pandas as pd

OTHER = "__other__"


class CategoricalEncoder:
    """One-hot các cột phân loại; giá trị hiếm/lạ gom về "__other__"."""

    def __init__(self, max_categories: int = 20) -> None:
        self.max_categories = max_categories
        self.columns_: list[str] = []
        self.categories_: dict[str, list[str]] = {}  # cột -> danh mục giữ lại (đã thêm OTHER)
        self.feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "CategoricalEncoder":
        self.columns_ = []
        self.categories_ = {}
        self.feature_names_ = []
        for c in cols:
            if c not in df.columns:
                continue
            counts = df[c].value_counts(dropna=False)
            # thứ tự ổn định: giảm dần theo tần suất, rồi theo tên giá trị
            kept = sorted(counts.index[: self.max_categories], key=lambda v: (-counts[v], str(v)))
            kept = [v for v in kept if pd.notna(v)]
            self.columns_.append(c)
            self.categories_[c] = [*kept, OTHER]
            self.feature_names_.extend(f"{c}={cat}" for cat in self.categories_[c])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in df.columns:
            if c in self.categories_:
                cats = self.categories_[c]
                cat_set = {cat for cat in cats if cat != OTHER}
                vals = df[c].map(lambda v: v if (pd.notna(v) and v in cat_set) else OTHER)
                for cat in cats:
                    out[f"{c}={cat}"] = (vals == cat).astype(float)
            else:
                out[c] = df[c]
        return out
