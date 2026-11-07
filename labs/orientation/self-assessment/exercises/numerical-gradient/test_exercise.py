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


numerical_grad = load().numerical_grad


def test_quadratic():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(4, 4))
    A = M @ M.T                                   # (4, 4) 대칭
    b = rng.normal(size=4)                        # (4,)
    f = lambda w: 0.5 * w @ A @ w + b @ w
    w = rng.normal(size=4)
    assert np.allclose(numerical_grad(f, w), A @ w + b, atol=1e-6), "이차식 0.5 wᵀAw + bᵀw의 gradient는 Aw + b입니다"


def test_sin_sum():
    w = np.array([0.0, 0.5, 1.0, -2.0])
    g = numerical_grad(lambda v: np.sin(v).sum(), w)
    assert np.allclose(g, np.cos(w), atol=1e-8), "Σ sin(w_i)의 gradient는 cos(w)입니다"


def test_mean_squared_error():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(30, 3)), rng.normal(size=30)
    f = lambda w: np.mean((X @ w - y) ** 2)
    w = rng.normal(size=3)
    assert np.allclose(numerical_grad(f, w), 2 / 30 * X.T @ (X @ w - y), atol=1e-6)


def test_shape_and_no_mutation():
    w = np.array([1.0, 2.0, 3.0])
    w_before = w.copy()
    g = numerical_grad(lambda v: (v ** 2).sum(), w)
    assert np.shape(g) == (3,), f"반환값의 shape은 w와 같은 (3,)이어야 합니다. 현재 {np.shape(g)}"
    assert np.array_equal(w, w_before), "numerical_grad가 입력 w를 바꾸었습니다"


def test_central_beats_forward():
    f = lambda v: (v ** 3).sum()
    w, h = np.array([1.0, -0.5]), 1e-3
    exact = 3 * w ** 2
    forward = np.array([(f(w + h * e) - f(w)) / h for e in np.eye(2)])
    central_err = np.abs(numerical_grad(f, w, h=h) - exact).max()
    forward_err = np.abs(forward - exact).max()
    assert central_err < forward_err / 100, "중심 차분이어야 합니다: (f(w+h e_i) - f(w-h e_i)) / (2h)"
