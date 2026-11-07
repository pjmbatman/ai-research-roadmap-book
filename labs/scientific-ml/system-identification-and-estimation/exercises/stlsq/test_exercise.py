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


stlsq = load().stlsq

A_TRUE = np.array([[-0.1, 2.0], [-2.0, -0.1]])  # 감쇠 진동자 dx/dt = A x


def library(X):  # (T, 2) → (T, 6): 1, x, y, xx, xy, yy
    x, y = X.T
    return np.column_stack([np.ones(len(X)), x, y, x * x, x * y, y * y])


def oscillator_data():
    t = np.linspace(0, 10, 2001)
    lam, V = np.linalg.eig(A_TRUE)
    c = np.linalg.solve(V, np.array([2.0, 0.0]))
    X = np.real((V * c) @ np.exp(np.outer(lam, t))).T  # (T, 2) 정확한 해
    return X, X @ A_TRUE.T                              # 정확한 미분


def test_recovers_oscillator():
    X, dX = oscillator_data()
    Xi = stlsq(library(X), dX, thresh=0.05)
    assert np.shape(Xi) == (6, 2), f"Xi의 shape은 (p, d) = (6, 2)여야 합니다. 현재 {np.shape(Xi)}"
    expected = np.zeros((6, 2))
    expected[1:3] = A_TRUE.T                            # x, y 행만 0이 아님
    assert np.allclose(Xi, expected, atol=1e-6), "참 계수 네 개만 남아야 합니다"


def test_exact_zeros():
    X, dX = oscillator_data()
    Xi = stlsq(library(X), dX, thresh=0.05)
    assert np.all(Xi[[0, 3, 4, 5]] == 0.0), "임계값보다 작은 계수는 정확히 0이어야 합니다"


def test_noise_spurious_terms_removed():
    rng = np.random.default_rng(0)
    X, dX = oscillator_data()
    Xi = stlsq(library(X), dX + 0.01 * rng.normal(size=dX.shape), thresh=0.05)
    assert np.count_nonzero(Xi) == 4, f"잡음이 조금 있어도 항 4개만 남아야 합니다. 현재 {np.count_nonzero(Xi)}개"
    assert np.allclose(Xi[1:3], A_TRUE.T, atol=0.01)


def test_large_threshold_kills_small_terms():
    X, dX = oscillator_data()
    Xi = stlsq(library(X), dX, thresh=0.5)
    assert Xi[1, 0] == 0.0 and Xi[2, 1] == 0.0, "감쇠 계수 -0.1은 임계값 0.5에서 잘려야 합니다"
    assert np.count_nonzero(Xi) == 2
