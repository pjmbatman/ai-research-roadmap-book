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


smooth = load().smooth


def test_hand_computed():
    # w = 0.5: s1 = 0.5*4 = 2 → 2/(1-0.5) = 4, s2 = 0.5*2 + 0.5*0 = 1 → 1/(1-0.25) = 4/3
    out = smooth([4.0, 0.0], 0.5)
    assert np.allclose(out, [4.0, 4.0 / 3.0]), "보정한 값은 [4, 4/3]이어야 합니다"


def test_first_value_is_not_pulled_to_zero():
    out = smooth([10.0, 9.0, 8.0], 0.99)
    assert out[0] == pytest.approx(10.0), "보정이 있으면 첫 값은 원본과 같습니다. 0.1이면 보정을 빠뜨린 것입니다"


def test_weight_zero_is_identity():
    x = np.random.default_rng(0).normal(size=20)
    assert np.allclose(smooth(x, 0.0), x)


def test_constant_input():
    assert np.allclose(smooth([0.3] * 7, 0.9), 0.3)


def test_nonfinite_values_pass_through():
    out = smooth([1.0, np.nan, 3.0, np.inf, 5.0], 0.5)
    assert np.isnan(out[1]) and np.isinf(out[3]), "NaN·inf는 그 자리에 원본 그대로 남아야 합니다"
    ref = smooth([1.0, 3.0, 5.0], 0.5)
    assert np.allclose(out[[0, 2, 4]], ref), "NaN·inf는 상태와 개수를 바꾸지 않아야 합니다"


def test_lag_after_step_change():
    x = np.r_[np.zeros(200), np.ones(200)]          # step 200에서 0 → 1
    out = smooth(x, 0.9)
    k = np.argmax(out[200:] >= 0.5)                 # 절반을 넘는 데 걸린 step
    assert 5 <= k <= 7, "w = 0.9이면 계단 변화의 절반을 따라가는 데 약 6 step이 걸립니다"
    out99 = smooth(x, 0.99)
    assert np.argmax(out99[200:] >= 0.5) > 60, "w가 클수록 지연이 길어집니다"


def test_returns_float_array_of_same_length():
    out = smooth([1, 2, 3], 0.6)
    assert isinstance(out, np.ndarray) and out.shape == (3,) and out.dtype.kind == "f"
