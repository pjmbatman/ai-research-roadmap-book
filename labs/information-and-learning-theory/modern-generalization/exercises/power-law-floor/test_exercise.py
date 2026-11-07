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


mod = load()
N = np.array([25, 50, 100, 200, 400, 800, 1600, 3200], dtype=float)
TRUE = 0.3 * N**-0.6 + 0.05


def test_known_floor_exact():
    a, b = mod.fit_power_law(N, TRUE, floor=0.05)
    assert np.isclose(a, 0.3) and np.isclose(b, -0.6), "바닥을 빼고 로그–로그 직선을 맞추면 a, b를 정확히 찾아야 합니다"


def test_no_floor_underpredicts():
    small = N <= 200
    a, b = mod.fit_power_law(N[small], TRUE[small], floor=0.0)
    pred = a * 3200**b
    assert pred < TRUE[-1], "바닥을 무시한 외삽은 실제보다 낮게 예측해야 합니다"
    assert b > -0.6, "바닥을 무시하면 기울기가 완만하게 추정됩니다"


def test_grid_floor_better_extrapolation():
    rng = np.random.default_rng(0)
    small = N <= 400
    L_obs = TRUE[small] * (1 + 0.01 * rng.normal(size=small.sum()))
    a, b, c = mod.fit_with_floor(N[small], L_obs, np.linspace(0, 0.06, 121))
    assert 0.03 < c < 0.06, f"바닥 추정이 참값 0.05 근처여야 합니다 (현재 {c})"
    a0, b0 = mod.fit_power_law(N[small], L_obs, 0.0)
    err_floor = abs(a * 3200**b + c - TRUE[-1])
    err_none = abs(a0 * 3200**b0 - TRUE[-1])
    assert err_floor < err_none, "바닥을 포함한 외삽이 더 정확해야 합니다"


def test_invalid_floor():
    with pytest.raises(ValueError):
        mod.fit_power_law(N, TRUE, floor=0.06)
