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
PTS = np.random.default_rng(0).uniform(-2, 2, size=(40, 2))   # (40, 2)


def test_quadratic_hessian():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    H = m.hessian_fd(lambda x: 0.5 * x @ A @ x, np.array([0.3, -0.7]))
    assert np.shape(H) == (2, 2), f"Hessian의 shape은 (n, n)이어야 합니다. 현재 {np.shape(H)}"
    assert np.allclose(H, A, atol=1e-4), "이차형식 x^T A x / 2의 Hessian은 A입니다"


def test_nonquadratic_hessian_3d():
    f = lambda x: np.sin(x[0]) * x[1] + x[2] ** 3
    x = np.array([0.4, 1.2, -0.5])
    exact = np.array([[-np.sin(0.4) * 1.2, np.cos(0.4), 0.0],
                      [np.cos(0.4), 0.0, 0.0],
                      [0.0, 0.0, 6 * -0.5]])
    H = m.hessian_fd(f, x)
    assert np.allclose(H, exact, atol=1e-4), "3차원 함수의 Hessian이 해석적 값과 다릅니다"
    assert np.allclose(H, np.transpose(H)), "Hessian은 대칭이어야 합니다"


@pytest.mark.parametrize("name,f,expected", [
    ("x^2 + 3y^2", lambda p: p[0] ** 2 + 3 * p[1] ** 2, True),
    ("log(1 + e^(x+y))", lambda p: np.log1p(np.exp(p[0] + p[1])), True),         # 볼록이지만 Hessian이 특이행렬
    ("x^2 - y^2", lambda p: p[0] ** 2 - p[1] ** 2, False),                          # 안장
    ("Rosenbrock", lambda p: (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2, False),
])
def test_is_convex_on(name, f, expected):
    assert bool(m.is_convex_on(f, PTS)) == expected, f"{name}: 기대한 판정은 {expected}입니다"


def test_single_counterexample_is_enough():
    f = lambda p: p[0] ** 4 / 4 - p[0] ** 2 / 2 + p[1] ** 2 / 2       # |x| < 1/sqrt(3)에서만 오목
    far = np.array([[2.0, 0.0], [-2.0, 1.0], [1.5, -1.0]])
    assert bool(m.is_convex_on(f, far)) is True, "이 점들에서는 Hessian이 PSD입니다"
    assert bool(m.is_convex_on(f, np.vstack([far, [[0.0, 0.0]]]))) is False, "한 점이라도 음의 고윳값이면 볼록이 아닙니다"
