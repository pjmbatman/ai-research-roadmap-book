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


lm_step = load().lm_step


def test_shape():
    rng = np.random.default_rng(0)
    d = lm_step(rng.normal(size=(7, 3)), rng.normal(size=7), 0.1)
    assert np.shape(d) == (3,), f"step의 shape은 (p,) = (3,)이어야 합니다. 현재 {np.shape(d)}"


def test_gauss_newton_solves_linear_least_squares():
    rng = np.random.default_rng(1)
    A, b = rng.normal(size=(20, 4)), rng.normal(size=20)
    theta = rng.normal(size=4)
    d = lm_step(A, A @ theta - b, 0.0)                      # 잔차 r = A theta - b, Jacobian = A
    assert np.allclose(theta + d, np.linalg.lstsq(A, b, rcond=None)[0]), \
        "lam = 0이면 선형 최소제곱 문제를 한 step에 풀어야 합니다(Gauss-Newton)"


def test_large_lambda_is_gradient_step():
    rng = np.random.default_rng(2)
    J, r = rng.normal(size=(10, 3)), rng.normal(size=10)
    g = J.T @ r
    d = lm_step(J, r, 1e8)
    assert np.allclose(d, -g / 1e8, rtol=1e-4), "lam이 크면 step은 -J^T r / lam(짧은 gradient step)에 가까워야 합니다"


def test_step_shrinks_and_decreases_model_cost():
    rng = np.random.default_rng(3)
    J, r = rng.normal(size=(12, 4)), rng.normal(size=12)
    norms = []
    for lam in [1e-3, 1e-1, 1e1, 1e3]:
        d = lm_step(J, r, lam)
        norms.append(np.linalg.norm(d))
        assert np.linalg.norm(r + J @ d) < np.linalg.norm(r), "선형 근사 r + J d의 크기가 줄어야 합니다"
    assert all(a > b for a, b in zip(norms, norms[1:])), "lam이 커질수록 step의 길이가 줄어야 합니다"
