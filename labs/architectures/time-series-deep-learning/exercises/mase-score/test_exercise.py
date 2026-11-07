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


mase = load().mase


def test_perfect_forecast_is_zero():
    rng = np.random.default_rng(0)
    y_train, y = rng.normal(size=50), rng.normal(size=10)
    assert mase(y, y, y_train) == 0.0, "예측이 정답과 같으면 0이어야 합니다"


def test_hand_computed():
    y_train = np.array([1.0, 3.0, 2.0, 4.0])              # |3-1|, |2-3|, |4-2| → 척도 5/3
    y_true, y_pred = np.array([5.0, 6.0]), np.array([4.0, 8.0])   # MAE 1.5
    assert np.isclose(mase(y_true, y_pred, y_train, m=1), 1.5 / (5 / 3)), "MAE / 척도로 계산해야 합니다"


def test_seasonal_period():
    y_train = np.array([0.0, 10.0, 1.0, 11.0, 3.0, 12.0])   # m=2 차이: 1, 1, 2, 1 → 척도 1.25
    y_true, y_pred = np.zeros(4), np.full(4, 2.5)
    assert np.isclose(mase(y_true, y_pred, y_train, m=2), 2.5 / 1.25), "m 시점 전 값과의 차이로 척도를 구해야 합니다"


def test_unit_invariant():
    rng = np.random.default_rng(1)
    y_train, y, p = rng.normal(size=100), rng.normal(size=(5, 24)), rng.normal(size=(5, 24))
    a = mase(y, p, y_train, m=24)
    b = mase(1000 * y + 3, 1000 * p + 3, 1000 * y_train + 3, m=24)   # 단위와 원점을 바꿈
    assert np.isclose(a, b), "단위를 바꿔도 MASE는 같아야 합니다"


def test_scale_uses_train_only():
    rng = np.random.default_rng(2)
    y_train = rng.normal(size=60)
    y, p = rng.normal(size=10), rng.normal(size=10)
    base = mase(y, p, y_train, m=1)
    assert np.isclose(mase(y, p, y_train, m=1), base)
    y_big = y.copy(); p_big = p.copy()
    y_big[0] += 100; p_big[0] += 100                        # 평가 구간 값이 커져도 오차가 같으면
    assert np.isclose(mase(y_big, p_big, y_train, m=1), base), "척도는 학습 구간만으로 계산해야 합니다"


def test_constant_train_raises():
    with pytest.raises(ValueError):
        mase(np.ones(3), np.zeros(3), np.full(10, 2.0), m=1)


def test_returns_float():
    out = mase(np.ones(3), np.zeros(3), np.arange(10.0))
    assert np.ndim(out) == 0, "float 하나를 반환해야 합니다"
