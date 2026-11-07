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


mod = load()


def simulate(phi, T, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros(T + 200)
    for t in range(len(phi), len(x)):
        x[t] = sum(phi[k] * x[t - 1 - k] for k in range(len(phi))) + rng.normal()
    return x[200:]                                                  # 초기 구간 버림


def test_recovers_ar2():
    x = simulate([0.6, -0.3], 5000, seed=0)
    phi = mod.fit_ar(x, 2)
    assert np.shape(phi) == (2,)
    assert np.allclose(phi, [0.6, -0.3], atol=0.05), f"AR(2) 계수 [0.6, -0.3]을 되찾아야 합니다. 현재 {np.round(phi, 3)}"


def test_fit_exact_on_noiseless_series():
    x = np.zeros(30)
    x[0], x[1] = 1.0, 0.5
    for t in range(2, 30):
        x[t] = 0.5 * x[t - 1] + 0.2 * x[t - 2]
    assert np.allclose(mod.fit_ar(x, 2), [0.5, 0.2]), "잡음이 없으면 계수를 정확히 되찾아야 합니다(lag 순서 확인)"


def test_recursive_forecast_by_hand():
    x = np.array([0.0, 1.0, 2.0])
    f = mod.forecast_ar(x, [0.5, 0.25], 3)
    x4 = 0.5 * 2.0 + 0.25 * 1.0          # 1.25
    x5 = 0.5 * x4 + 0.25 * 2.0           # 1.125
    x6 = 0.5 * x5 + 0.25 * x4            # 0.875
    assert np.allclose(f, [x4, x5, x6]), f"재귀 예측은 [{x4}, {x5}, {x6}]이어야 합니다. 현재 {np.round(f, 4)}"


def test_ar1_forecast_is_power():
    x = np.array([3.0, -1.0, 2.0])
    f = mod.forecast_ar(x, [0.8], 5)
    assert np.allclose(f, 2.0 * 0.8 ** np.arange(1, 6)), "AR(1)의 h 시점 예측은 phi^h * x_T입니다"


def test_long_horizon_reverts_to_mean():
    x = simulate([0.6, -0.3], 500, seed=1)
    f = mod.forecast_ar(x, mod.fit_ar(x, 2), 60)
    assert np.shape(f) == (60,)
    assert abs(f[-1]) < 1e-3, "정상 AR의 먼 미래 예측은 평균(0)으로 수렴해야 합니다"
