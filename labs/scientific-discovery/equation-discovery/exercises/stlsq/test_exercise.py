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


def poly_library(X):                                     # (T, 2) -> (T, 6): 1, x, y, x^2, xy, y^2
    x, y = X[:, 0], X[:, 1]
    return np.stack([np.ones_like(x), x, y, x**2, x * y, y**2], axis=1)


def sparse_system(seed=0, T=400):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.5, 3.0, size=(T, 2))               # 상태 공간을 넓게 덮는 표본
    Xi_true = np.zeros((6, 2))
    Xi_true[[1, 4], 0] = [1.0, -0.5]                     # dx/dt = x - 0.5 xy
    Xi_true[[2, 4], 1] = [-0.6, 0.2]                     # dy/dt = -0.6 y + 0.2 xy
    return poly_library(X), poly_library(X) @ Xi_true, Xi_true


def test_exact_recovery():
    Theta, dX, Xi_true = sparse_system()
    Xi = stlsq(Theta, dX, thr=0.05)
    assert np.shape(Xi) == (6, 2), f"반환 shape은 (p, n) = (6, 2)여야 합니다. 현재 {np.shape(Xi)}"
    assert np.allclose(Xi, Xi_true, atol=1e-8), "잡음 없는 데이터에서는 참 계수를 정확히 복원해야 합니다"


def test_threshold_respected():
    Theta, dX, _ = sparse_system(seed=1)
    rng = np.random.default_rng(2)
    dX = dX + 0.05 * rng.normal(size=dX.shape)           # 잡음이 있으면 작은 가짜 계수가 생긴다
    Xi = stlsq(Theta, dX, thr=0.15)
    nz = Xi[Xi != 0]
    assert np.all(np.abs(nz) >= 0.15), "0이 아닌 계수의 절댓값은 모두 임계값 이상이어야 합니다"


def test_all_small_column_becomes_zero():
    Theta, dX, _ = sparse_system(seed=3)
    dX = np.column_stack([dX[:, 0], 1e-3 * dX[:, 1]])    # 둘째 열은 모든 계수가 아주 작다
    Xi = stlsq(Theta, dX, thr=0.05)
    assert np.all(Xi[:, 1] == 0), "모든 계수가 임계값보다 작은 열은 전부 0이어야 합니다"
    assert np.count_nonzero(Xi[:, 0]) == 2, "첫째 열은 두 항(x, xy)이 남아야 합니다"


def test_refit_after_threshold():
    rng = np.random.default_rng(4)
    a = rng.normal(size=500)
    b = 0.9 * a + 0.1 * rng.normal(size=500)             # a와 강하게 상관된 후보
    Theta = np.column_stack([a, b])
    dX = (2.0 * a + 0.03 * b)[:, None]                   # b의 참 계수 0.03은 임계값 아래
    Xi = stlsq(Theta, dX, thr=0.1)
    assert Xi[1, 0] == 0, "작은 계수는 0으로 잘려야 합니다"
    lone = np.linalg.lstsq(Theta[:, :1], dX[:, 0], rcond=None)[0][0]
    assert np.isclose(Xi[0, 0], lone), "자른 뒤에는 남은 항만으로 다시 최소제곱을 풀어야 합니다"
