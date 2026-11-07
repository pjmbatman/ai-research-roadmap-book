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


euler_with_sensitivity = load().euler_with_sensitivity


def euler(r, K, y0, h, steps):
    y = [y0]
    for _ in range(steps):
        y.append(y[-1] + h * r * y[-1] * (1 - y[-1] / K))
    return np.array(y)


def test_shapes_and_initial():
    y, s = euler_with_sensitivity(1.0, 10.0, 0.5, 0.1, 20)
    assert np.shape(y) == (21,) and np.shape(s) == (21,), "y, s의 shape은 (steps + 1,)이어야 합니다"
    assert y[0] == 0.5 and s[0] == 0.0, "첫 원소는 y0과 0이어야 합니다"


@pytest.mark.parametrize("r,K,y0,h,steps", [(1.0, 10.0, 0.5, 0.1, 50), (2.5, 1.0, 0.05, 0.02, 200), (0.3, 5.0, 4.0, 0.5, 30)])
def test_matches_finite_difference(r, K, y0, h, steps):
    y, s = euler_with_sensitivity(r, K, y0, h, steps)
    assert np.allclose(y, euler(r, K, y0, h, steps)), "y가 explicit Euler의 결과와 다릅니다"
    d = 1e-6
    fd = (euler(r + d, K, y0, h, steps) - euler(r - d, K, y0, h, steps)) / (2 * d)
    assert np.allclose(s, fd, rtol=1e-5, atol=1e-8), "민감도가 같은 Euler 시뮬레이션의 중심차분과 다릅니다"


def test_zero_rate():
    y, s = euler_with_sensitivity(0.0, 10.0, 2.0, 0.1, 10)
    assert np.allclose(y, 2.0), "r = 0이면 y는 변하지 않아야 합니다"
    assert np.allclose(s, 0.1 * np.arange(11) * 2.0 * (1 - 0.2)), "r = 0이면 s_n = n h y0 (1 - y0/K)입니다"


def test_equilibrium():
    y, s = euler_with_sensitivity(1.7, 3.0, 3.0, 0.1, 25)
    assert np.allclose(y, 3.0) and np.allclose(s, 0.0), "평형점 y0 = K에서는 y가 변하지 않고 민감도도 0입니다"
