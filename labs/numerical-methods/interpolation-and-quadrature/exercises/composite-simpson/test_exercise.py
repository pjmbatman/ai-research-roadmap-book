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


composite_simpson = load().composite_simpson


@pytest.mark.parametrize("n", [2, 4, 10])
def test_exact_for_cubics(n):
    f = lambda x: 4 * x**3 - 3 * x**2 + 2 * x - 1
    F = lambda x: x**4 - x**3 + x**2 - x            # 원시함수
    a, b = -1.0, 2.0
    got = composite_simpson(f, a, b, n)
    assert np.isclose(got, F(b) - F(a), rtol=0, atol=1e-12), f"3차 다항식은 정확해야 합니다: {got} != {F(b) - F(a)}"


def test_sin_value():
    got = composite_simpson(np.sin, 0.0, np.pi, 10)
    assert abs(got - 2.0) < 2e-4, f"∫_0^π sin x dx = 2, n=10에서 오차가 너무 큽니다 (결과 {got})"


def test_fourth_order_convergence():
    errs = [abs(composite_simpson(np.sin, 0.0, np.pi, n) - 2.0) for n in (8, 16, 32)]
    ratios = [errs[0] / errs[1], errs[1] / errs[2]]
    for r in ratios:
        assert 14.0 < r < 18.0, f"n을 두 배로 하면 오차가 약 16배 줄어야 합니다. 비율 {ratios}"


def test_returns_float_and_reversed_interval():
    got = composite_simpson(np.exp, 1.0, 0.0, 8)
    assert isinstance(got, float), "반환값은 float여야 합니다"
    assert np.isclose(got, -(np.e - 1.0), atol=1e-5), "b < a이면 부호가 바뀐 값이 나와야 합니다"


@pytest.mark.parametrize("n", [1, 3, 7])
def test_odd_n_raises(n):
    with pytest.raises(ValueError):
        composite_simpson(np.sin, 0.0, 1.0, n)
