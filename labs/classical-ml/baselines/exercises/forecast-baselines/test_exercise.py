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


forecast_baselines = load().forecast_baselines
KEYS = {"mean", "persistence", "seasonal", "drift"}


def test_hand_example():
    out = forecast_baselines([1.0, 2.0, 5.0, 6.0, 7.0], horizon=5, period=3)
    assert set(out) == KEYS, f"키는 {sorted(KEYS)}여야 합니다. 현재 {sorted(out)}"
    assert np.allclose(out["mean"], 4.2), "mean은 학습 시계열의 평균 4.2를 반복해야 합니다"
    assert np.allclose(out["persistence"], 7.0), "persistence는 마지막 값 7을 반복해야 합니다"
    assert np.allclose(out["seasonal"], [5, 6, 7, 5, 6]), "seasonal은 마지막 한 주기 [5, 6, 7]을 반복해야 합니다"
    assert np.allclose(out["drift"], [8.5, 10.0, 11.5, 13.0, 14.5]), "drift의 기울기는 (y_T - y_1)/(T - 1) = 1.5입니다"


def test_shapes_and_list_input():
    out = forecast_baselines(list(range(10)), horizon=7, period=4)
    for k in KEYS:
        assert np.shape(out[k]) == (7,), f"{k}의 shape은 (horizon,) = (7,)이어야 합니다. 현재 {np.shape(out[k])}"


def test_seasonal_wins_on_periodic_signal():
    t = np.arange(96)
    y = 10 + 3 * np.sin(2 * np.pi * t / 24)                 # 주기 24
    out = forecast_baselines(y[:72], horizon=24, period=24)
    err = {k: np.abs(v - y[72:]).mean() for k, v in out.items()}
    assert err["seasonal"] < 1e-9, "잡음 없는 주기 신호에서 seasonal 예측은 정확해야 합니다"
    assert min(err, key=err.get) == "seasonal"


def test_drift_wins_on_linear_trend():
    y = 2.0 + 0.5 * np.arange(60)
    out = forecast_baselines(y[:40], horizon=20, period=10)
    err = {k: np.abs(v - y[40:]).mean() for k, v in out.items()}
    assert err["drift"] < 1e-9, "직선 추세에서 drift 예측은 정확해야 합니다"
    assert err["persistence"] < err["mean"], "추세가 있으면 최근값이 평균보다 낫습니다"
