"""BaseRegressor: giao diện chung cho mọi mô hình (fit/predict/get_params/set_params/clone).

Ngoài ra còn hỗ trợ lưu/nạp mô hình dưới dạng file .npy của numpy qua save_model/load_model:
trạng thái học được (`get_state`) chỉ gồm các mảng numpy + tham số thuần Python.
"""
import inspect
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

_REGISTRY: dict[str, type["BaseRegressor"]] = {}


class BaseRegressor(ABC):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        _REGISTRY[cls.__name__] = cls

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseRegressor":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def get_params(self) -> dict:
        """Trả các tham số truyền vào __init__ (đọc thuộc tính cùng tên)."""
        sig = inspect.signature(self.__init__)
        params = {}
        for name, p in sig.parameters.items():
            if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            params[name] = getattr(self, name, p.default)
        return params

    def set_params(self, **params) -> "BaseRegressor":
        valid = set(inspect.signature(self.__init__).parameters) - {"self"}
        for k, v in params.items():
            if k not in valid:
                raise ValueError(f"Tham số lạ: {k!r} (hợp lệ: {sorted(valid)})")
            setattr(self, k, v)
        return self

    def clone(self) -> "BaseRegressor":
        """Mô hình mới cùng lớp, cùng tham số, CHƯA fit."""
        return type(self)(**self.get_params())

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    # --- lưu/nạp dạng .npy ---
    def get_state(self) -> dict:
        """Trả dict {"class", "params", "learned"} — phần learned chỉ chứa mảng numpy/scalar."""
        return {"class": type(self).__name__, "params": self.get_params(), "learned": self._learned_state()}

    def _learned_state(self) -> dict:
        raise NotImplementedError

    def _set_learned_state(self, learned: dict) -> None:
        raise NotImplementedError

    @classmethod
    def from_state(cls, state: dict) -> "BaseRegressor":
        model = cls(**state["params"])
        model._set_learned_state(state["learned"])
        return model


def save_model(model: BaseRegressor, path) -> None:
    """Lưu mô hình ra file .npy của numpy (np.save tự thêm đuôi .npy nếu thiếu)."""
    path = Path(path)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, model.get_state(), allow_pickle=True)


def load_model(path) -> BaseRegressor:
    """Nạp mô hình từ file .npy đã lưu bằng save_model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {path}")
    state = np.load(path, allow_pickle=True).item()
    # import lồng trong hàm để các lớp con tự đăng ký vào _REGISTRY (tránh circular import ở import time)
    from src.models import decision_tree, gradient_boosting, linear_regression, random_forest  # noqa: F401

    cls = _REGISTRY.get(state.get("class"))
    if cls is None:
        raise ValueError(f"File {path} không chứa mô hình hợp lệ (class={state.get('class')!r})")
    return cls.from_state(state)
