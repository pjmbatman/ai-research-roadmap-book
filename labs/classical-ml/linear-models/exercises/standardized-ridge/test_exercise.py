import importlib.util
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fit_ridge = load().fit_ridge


def data(seed=0, n=40, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * np.array([1.0, 50.0, 0.01])[:p] + np.array([5.0, 300.0, 0.0])[:p]
    y = 2.0 + X @ np.array([0.5, 0.02, 80.0])[:p] + rng.normal(0, 0.1, n)
    return X, y


def test_shapes():
    X, y = data()
    w, b = fit_ridge(X, y, 1.0)
    assert np.shape(w) == (3,), f"w의 shape은 (p,) = (3,)이어야 합니다. 현재 {np.shape(w)}"
    assert np.ndim(b) == 0, "b는 스칼라여야 합니다"


def test_lam_zero_is_ols():
    X, y = data(1)
    w, b = fit_ridge(X, y, 0.0)
    coef = np.linalg.lstsq(np.column_stack([X, np.ones(len(y))]), y, rcond=None)[0]
    assert np.allclose(w, coef[:3]) and np.isclose(b, coef[3]), "lam = 0이면 절편 있는 최소제곱과 같아야 합니다"


def test_unit_change_does_not_change_predictions():
    X, y = data(2)
    X2 = X.copy()
    X2[:, 1] *= 1000.0                                      # 두 번째 특징의 단위를 바꾼다
    w1, b1 = fit_ridge(X, y, 5.0)
    w2, b2 = fit_ridge(X2, y, 5.0)
    assert np.allclose(X @ w1 + b1, X2 @ w2 + b2), "표준화했다면 단위를 바꿔도 예측이 같아야 합니다"
    assert np.isclose(w2[1], w1[1] / 1000.0), "원래 단위의 계수는 단위 변환만큼 바뀌어야 합니다"


def test_intercept_not_penalized():
    X, y = data(3)
    w, b = fit_ridge(X, y, 1e9)
    assert np.allclose(w * X.std(0), 0, atol=1e-6), "λ가 매우 크면 (표준화된) 계수는 0에 가까워야 합니다"
    assert np.isclose(b, y.mean(), atol=1e-4), f"절편은 벌점을 받지 않으므로 y의 평균 {y.mean():.4f}에 가까워야 합니다. 현재 {b:.4f}"


def test_optimality_condition():
    X, y = data(4)
    lam = 3.0
    w, b = fit_ridge(X, y, lam)
    mu, s = X.mean(0), X.std(0)
    Z, v = (X - mu) / s, w * s
    assert np.allclose(Z.T @ (y - y.mean() - Z @ v), lam * v), "표준화된 공간에서 Z^T(ỹ - Z v) = λ v를 만족해야 합니다"
