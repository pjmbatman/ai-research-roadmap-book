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
one_step, rollout = mod.one_step, mod.rollout

R, OMEGA, T = 0.98, 0.3, 200
W_TRUE = np.array([2 * R * np.cos(OMEGA), -R ** 2])
SERIES = R ** np.arange(T) * np.cos(OMEGA * np.arange(T))     # 감쇠 진동, AR(2)가 정확히 만든다


def test_rollout_reproduces_damped_oscillation():
    out = rollout(W_TRUE, SERIES[:2], T)
    assert np.shape(out) == (T,), f"shape은 (T,) = ({T},)이어야 합니다"
    assert np.allclose(out, SERIES), "정확한 계수로 rollout하면 감쇠 진동 전체가 나와야 합니다"


def test_one_step_exact_model():
    out = one_step(W_TRUE, SERIES)
    assert np.shape(out) == (T,)
    assert np.allclose(out, SERIES), "정확한 계수의 one-step 예측은 참값과 같아야 합니다"


def test_first_values_untouched():
    x = np.arange(10.0)
    w = np.array([0.5, 0.2, 0.1])
    assert np.allclose(one_step(w, x)[:3], x[:3]), "one-step의 앞 p개는 x를 그대로 둡니다"
    assert np.allclose(rollout(w, x[:3], 10)[:3], x[:3]), "rollout의 앞 p개는 x_init입니다"


def test_lag_order():
    x = np.array([1.0, 2.0, 4.0, 0.0])
    out = one_step(np.array([1.0, 0.0]), x)
    assert np.isclose(out[2], 2.0), "w[0]은 바로 전 시점(x[t-1])의 계수입니다"
    out = one_step(np.array([0.0, 1.0]), x)
    assert np.isclose(out[2], 1.0), "w[1]은 두 시점 전(x[t-2])의 계수입니다"


def test_one_step_uses_truth_rollout_uses_predictions():
    rng = np.random.default_rng(0)
    noisy = SERIES + 0.01 * rng.normal(size=T)
    w = np.array([1.0])                                        # AR(1): 다음 값 = 지금 값 (persistence)
    assert np.allclose(one_step(w, noisy)[1:], noisy[:-1]), "one-step은 매번 참값 과거를 써야 합니다"
    assert np.allclose(rollout(w, noisy[:1], T), noisy[0]), "rollout은 자기 예측을 써야 합니다(상수 계열)"


def test_error_accumulates_in_rollout():
    w_bad = W_TRUE * np.array([1.0, 0.99])                     # 계수 하나가 1% 틀린 모델
    e1 = np.abs(one_step(w_bad, SERIES) - SERIES).max()
    er = np.abs(rollout(w_bad, SERIES[:2], T) - SERIES).max()
    assert er > 5 * e1, "계수가 조금 틀린 모델의 rollout 오차는 one-step 오차보다 훨씬 커야 합니다"
