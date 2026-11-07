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


rk4_step = load().rk4_step


@pytest.mark.parametrize("h", [0.1, 0.5, 1.0])
def test_exponential_taylor(h):
    got = rk4_step(lambda t, y: y, 0.0, np.array([1.0]), h)
    expect = 1 + h + h**2 / 2 + h**3 / 6 + h**4 / 24
    assert np.shape(got) == (1,), f"반환 shape은 입력과 같은 (1,)이어야 합니다. 현재 {np.shape(got)}"
    assert np.isclose(got[0], expect, rtol=1e-14), f"y'=y 한 step: {got[0]} != {expect}"


def test_time_dependence():
    # y' = t^3, y(1) = 0 → y(1 + h) = ((1+h)^4 - 1) / 4. RK4는 이 경우 Simpson 규칙이라 정확하다.
    h = 0.5
    got = rk4_step(lambda t, y: np.array([t**3]), 1.0, [0.0], h)
    assert np.isclose(got[0], ((1 + h) ** 4 - 1) / 4, rtol=1e-13), "f에 각 단계의 시각 t, t+h/2, t+h를 넘겨야 합니다"


def test_vector_and_no_mutation():
    y0 = np.array([1.0, 0.0])
    before = y0.copy()
    y1 = rk4_step(lambda t, y: np.array([y[1], -y[0]]), 0.0, y0, 0.1)
    assert np.allclose(y0, before), "입력 y를 제자리에서 바꾸면 안 됩니다"
    assert np.allclose(y1, [np.cos(0.1), -np.sin(0.1)], atol=1e-7), "조화 진동자 한 step 결과가 정확해와 너무 다릅니다"


def test_global_fourth_order():
    f = lambda t, y: np.array([y[1], -y[0]])
    errs = []
    for n in (20, 40, 80):
        h, y = 2 * np.pi / n, np.array([1.0, 0.0])
        for i in range(n):
            y = rk4_step(f, i * h, y, h)
        errs.append(np.linalg.norm(y - [1.0, 0.0]))
    ratios = [errs[0] / errs[1], errs[1] / errs[2]]
    assert all(14 < r < 18 for r in ratios), f"h를 절반으로 하면 오차가 약 16배 줄어야 합니다. 비율 {ratios}"
