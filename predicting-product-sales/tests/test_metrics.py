"""Test cho src/metrics.py: giá trị tính tay, trường hợp biên, kiểm tra lỗi."""
import numpy as np
import pytest

from src.metrics import evaluate_all, mae, mse, r2, rmse


def test_hand_computed_values():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 5]
    assert mse(y_true, y_pred) == pytest.approx(4 / 3)
    assert mae(y_true, y_pred) == pytest.approx(2 / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(4 / 3))


def test_r2_perfect_prediction():
    y = [1.0, 2.0, 3.0, 4.0]
    assert r2(y, y) == pytest.approx(1.0)


def test_r2_mean_prediction_is_zero():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [float(np.mean(y_true))] * 4  # dự đoán bằng trung bình -> r2 = 0
    assert r2(y_true, y_pred) == pytest.approx(0.0)


def test_r2_zero_variance_returns_0():
    y_true = [5.0, 5.0, 5.0]
    assert r2(y_true, [5.0, 5.0, 4.0]) == 0.0  # phương sai y_true = 0 -> 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mse([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        rmse([], [])


def test_evaluate_all_keys():
    out = evaluate_all([1, 2, 3], [1, 2, 5])
    assert set(out.keys()) == {"mse", "mae", "rmse", "r2"}
    assert all(isinstance(v, float) for v in out.values())
