import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()


def problem(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(40, 15))                       # (N, D)
    w_true = np.zeros(15)
    w_true[[1, 6, 9]] = [2.0, -1.5, 1.0]
    return A, A @ w_true + 0.1 * rng.normal(size=40)


def test_soft_threshold_values():
    out = m.soft_threshold(np.array([3.0, -0.5, 0.2, -2.0, 0.0]), 1.0)
    assert np.allclose(out, [2.0, 0.0, 0.0, -1.0, 0.0]), "|v| <= t 이면 0, 아니면 크기를 t만큼 줄입니다"


def test_soft_threshold_is_prox():
    # prox_t(v) = argmin_x 0.5 (x - v)^2 + t |x| 를 격자 탐색과 비교
    grid = np.linspace(-5, 5, 100001)
    for v in [2.3, -0.4, -3.1]:
        x_grid = grid[np.argmin(0.5 * (grid - v) ** 2 + 0.7 * np.abs(grid))]
        assert np.isclose(m.soft_threshold(np.array([v]), 0.7)[0], x_grid, atol=1e-4)


@pytest.mark.parametrize("lam", [1.0, 5.0, 20.0])
def test_lasso_optimality(lam):
    A, y = problem()
    w = np.asarray(m.ista(A, y, lam))
    r = A.T @ (y - A @ w)                               # (D,)
    nz = w != 0
    assert np.allclose(r[nz], lam * np.sign(w[nz]), atol=1e-4), "0이 아닌 좌표에서는 r_i = lam * sign(w_i)여야 합니다"
    assert np.all(np.abs(r[~nz]) <= lam + 1e-4), "0인 좌표에서는 |r_i| <= lam이어야 합니다"


def test_sparsity_recovers_support():
    A, y = problem()
    w = np.asarray(m.ista(A, y, 5.0))
    assert set(np.flatnonzero(w)) == {1, 6, 9}, "참 모델의 0이 아닌 좌표 {1, 6, 9}를 찾아야 합니다"


def test_large_lambda_gives_zero():
    A, y = problem()
    lam_max = np.max(np.abs(A.T @ y))
    assert np.all(np.asarray(m.ista(A, y, lam_max * 1.01)) == 0), "lam >= ||A^T y||_inf 이면 해는 0입니다"
