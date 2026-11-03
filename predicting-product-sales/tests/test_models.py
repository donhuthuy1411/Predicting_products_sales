"""Test cho 4 mô hình tự cài và BaseRegressor trên dữ liệu giả có đáp án biết trước."""
import numpy as np
import pytest

from src.metrics import rmse
from src.models.base import BaseRegressor, load_model, save_model
from src.models.decision_tree import DecisionTreeRegressor
from src.models.gradient_boosting import GradientBoostingRegressor
from src.models.linear_regression import LinearRegression
from src.models.random_forest import RandomForestRegressor


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(500, 4))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.1 * rng.normal(size=500)
    return X[:400], y[:400], X[400:], y[400:]


def test_linear_regression_coefficients(synthetic):
    X_tr, y_tr, X_te, y_te = synthetic
    m = LinearRegression(alpha=0.0).fit(X_tr, y_tr)
    assert np.allclose(m.coef_, [3, -2, 0, 0], atol=0.1)
    assert rmse(y_te, m.predict(X_te)) < 0.5 * np.std(y_te)


@pytest.mark.parametrize(
    "make_model",
    [
        lambda: DecisionTreeRegressor(max_depth=8, random_state=0),
        lambda: RandomForestRegressor(n_estimators=5, max_depth=8, max_features="sqrt", random_state=0),
        lambda: GradientBoostingRegressor(n_estimators=20, random_state=0),
    ],
)
def test_tree_models_fit_predict(synthetic, make_model):
    X_tr, y_tr, X_te, y_te = synthetic
    m = make_model().fit(X_tr, y_tr)
    pred = m.predict(X_te)
    assert pred.shape == (100,)
    assert rmse(y_te, pred) < 0.5 * np.std(y_te)


@pytest.mark.parametrize(
    "make_model",
    [
        lambda: LinearRegression(),
        lambda: DecisionTreeRegressor(),
        lambda: RandomForestRegressor(n_estimators=2),
        lambda: GradientBoostingRegressor(n_estimators=2),
    ],
)
def test_predict_before_fit_raises(make_model):
    with pytest.raises(RuntimeError):
        make_model().predict(np.zeros((3, 4)))


def test_clone_and_set_params(synthetic):
    X_tr, y_tr, _, _ = synthetic
    m = RandomForestRegressor(n_estimators=3, max_depth=4, random_state=7)
    c = m.clone()
    assert isinstance(c, RandomForestRegressor)
    assert c.get_params() == m.get_params()
    assert not hasattr(c, "estimators_")  # clone chưa fit
    m.set_params(n_estimators=9)
    assert m.n_estimators == 9
    with pytest.raises(ValueError):
        m.set_params(khong_co=1)


def test_same_random_state_same_prediction(synthetic):
    X_tr, y_tr, X_te, _ = synthetic
    for make in (
        lambda: RandomForestRegressor(n_estimators=5, random_state=123),
        lambda: GradientBoostingRegressor(n_estimators=10, random_state=123),
    ):
        p1 = make().fit(X_tr, y_tr).predict(X_te)
        p2 = make().fit(X_tr, y_tr).predict(X_te)
        assert np.allclose(p1, p2)


def test_base_is_abstract():
    with pytest.raises(TypeError):
        BaseRegressor()


@pytest.mark.parametrize(
    "make_model",
    [
        lambda: LinearRegression(alpha=1.0),
        lambda: DecisionTreeRegressor(max_depth=4, random_state=0),
        lambda: RandomForestRegressor(n_estimators=3, max_depth=4, random_state=0),
        lambda: GradientBoostingRegressor(n_estimators=5, random_state=0),
    ],
)
def test_save_load_npy_roundtrip(synthetic, tmp_path, make_model):
    X_tr, y_tr, X_te, _ = synthetic
    m = make_model().fit(X_tr, y_tr)
    path = tmp_path / f"{type(m).__name__}.npy"
    save_model(m, path)
    assert path.exists() and path.suffix == ".npy"
    m2 = load_model(path)
    assert type(m2) is type(m)
    assert m2.get_params() == m.get_params()
    assert np.allclose(m.predict(X_te), m2.predict(X_te))
