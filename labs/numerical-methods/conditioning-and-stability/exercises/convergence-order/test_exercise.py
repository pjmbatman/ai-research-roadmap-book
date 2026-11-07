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


estimate_order = load().estimate_order


def test_shape():
    p = estimate_order(np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]))
    assert np.shape(p) == (3,), f"길이 n-2 = 3인 배열이어야 합니다. 현재 shape {np.shape(p)}"


def test_linear():
    e = 0.5 ** np.arange(10)
    assert np.allclose(estimate_order(e), 1.0), "e_k = 0.5^k는 선형 수렴(p = 1)입니다"


def test_quadratic():
    e = [1e-1]
    for _ in range(4):
        e.append(e[-1] ** 2)
    assert np.allclose(estimate_order(np.array(e)), 2.0), "e_{k+1} = e_k^2는 2차 수렴(p = 2)입니다"


def test_newton_sqrt2():
    x, root, e = 3.0, np.sqrt(2.0), []
    for _ in range(5):
        e.append(abs(x - root))
        x = 0.5 * (x + 2.0 / x)            # Newton 반복으로 sqrt(2)
    p = estimate_order(np.array(e))
    assert abs(p[-1] - 2.0) < 0.1, f"Newton 반복의 마지막 추정값이 2에 가까워야 합니다. 현재 {p}"


def test_secant_golden_ratio():
    f = lambda x: x**3 - 2.0
    root = 2.0 ** (1 / 3)
    x0, x1, e = 1.0, 2.0, []
    for _ in range(7):                     # 오차가 1e-10 근처일 때까지
        e.append(abs(x1 - root))
        x0, x1 = x1, x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
    p = estimate_order(np.array(e))
    golden = (1 + np.sqrt(5)) / 2
    assert abs(p[-1] - golden) < 0.15, f"secant 반복의 추정값이 약 1.618이어야 합니다. 현재 {p}"
