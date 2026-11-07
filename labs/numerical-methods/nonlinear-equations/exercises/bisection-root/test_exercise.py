import importlib.util
import math
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


bisection = load().bisection
ROOT = 2.0945514815423265                      # x^3 - 2x - 5 = 0의 실근


def test_finds_root_within_xtol():
    x, _ = bisection(lambda x: x**3 - 2 * x - 5, 2.0, 3.0, xtol=1e-10)
    assert abs(x - ROOT) <= 1e-10, f"|x - 근| = {abs(x - ROOT):.1e}가 xtol보다 큽니다"


@pytest.mark.parametrize("xtol", [1e-3, 1e-6, 1e-12])
def test_iteration_count(xtol):
    a, b = 2.0, 3.0
    _, k = bisection(lambda x: x**3 - 2 * x - 5, a, b, xtol=xtol)
    expected = math.ceil(math.log2((b - a) / (2 * xtol)))
    assert k == expected, f"폭 (b-a)/2^k <= 2*xtol이 되는 첫 k = {expected}번이어야 합니다. 현재 {k}번"


def test_same_sign_raises():
    with pytest.raises(ValueError):
        bisection(lambda x: x**2 + 1, -1.0, 1.0)


def test_endpoint_is_root():
    x, k = bisection(lambda x: x - 1.0, 1.0, 3.0)
    assert x == 1.0 and k == 0


def test_decreasing_function_and_negative_interval():
    x, _ = bisection(lambda x: np.exp(-x) - 2.0, -5.0, 0.0, xtol=1e-12)
    assert abs(x - (-np.log(2.0))) <= 1e-12


def test_no_overflow_in_sign_test():
    f = lambda x: 1e200 * (x - 0.3)          # f(a) * f(m)는 overflow할 수 있다
    x, _ = bisection(f, 0.0, 1.0, xtol=1e-12)
    assert abs(x - 0.3) <= 1e-12
