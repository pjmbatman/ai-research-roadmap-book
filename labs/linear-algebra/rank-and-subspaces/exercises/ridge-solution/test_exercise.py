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


ridge = load().ridge


def collinear_data(seed=0):
    rng = np.random.default_rng(seed)
    temp_c = rng.uniform(10, 40, size=40)
    X = np.column_stack([np.ones(40), temp_c, 1.8 * temp_c + 32.0])   # (40, 3), rank 2
    y = 0.5 * temp_c + rng.normal(scale=0.3, size=40)
    return X, y


def test_normal_equations():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(30, 4)), rng.normal(size=30)
    w = np.asarray(ridge(X, y, 0.7))
    assert w.shape == (4,), f"w의 shape은 (D,) = (4,)여야 합니다. 현재 {w.shape}"
    assert np.allclose((X.T @ X + 0.7 * np.eye(4)) @ w, X.T @ y), "(X^T X + lam I) w = X^T y를 만족해야 합니다"


def test_duplicate_columns_get_equal_weights():
    rng = np.random.default_rng(1)
    x = rng.normal(size=25)
    X = np.column_stack([x, x])                                       # 똑같은 열 두 개
    w = np.asarray(ridge(X, 3.0 * x, 1e-3))
    assert np.isclose(w[0], w[1]), f"똑같은 두 열의 계수는 같아야 합니다. 현재 {w}"
    assert np.isclose(w.sum(), 3.0, atol=1e-3), f"두 계수의 합은 약 3이어야 합니다. 현재 {w.sum()}"


def test_shrinks_with_lambda():
    X, y = collinear_data()
    norms = [np.linalg.norm(ridge(X, y, lam)) for lam in [1e-3, 1.0, 1e2, 1e4]]
    assert all(a > b for a, b in zip(norms, norms[1:])), f"lam이 커질수록 |w|가 줄어야 합니다: {np.round(norms, 4)}"


def test_small_lambda_gives_min_norm_solution():
    X, y = collinear_data(3)
    w_min = np.linalg.lstsq(X, y)[0]
    w = np.asarray(ridge(X, y, 1e-6))
    assert np.allclose(w, w_min, atol=1e-3), f"lam → 0이면 최소 노름 해 {w_min}에 가까워야 합니다. 현재 {w}"
    n_unit = np.array([32.0, 1.8, -1.0]) / np.linalg.norm([32.0, 1.8, -1.0])
    assert abs(w @ n_unit) < 1e-3, "최소 노름 해는 null space 방향 성분이 0입니다"
