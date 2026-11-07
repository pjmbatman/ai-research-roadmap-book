import importlib.util
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load()
R0, R1, C1, DT = 0.020, 0.015, 2000.0, 1.0                                       # τ = 30 s
I_PULSE = np.where((np.arange(400) >= 50) & (np.arange(400) < 250), 3.0, 0.0)   # (400,) 3 A 펄스 200 s


def test_shape():
    eta = mod.overpotential(R0, R1, C1, I_PULSE, DT)
    assert np.shape(eta) == (400,), f"과전압의 shape은 (T,) = (400,)여야 합니다. 현재 {np.shape(eta)}"


def test_instant_jump_is_r0():
    eta = mod.overpotential(R0, R1, C1, I_PULSE, DT)
    assert np.isclose(eta[49], 0.0), "펄스 전에는 과전압이 0이어야 합니다"
    assert np.isclose(eta[50], 3.0 * R0), "펄스가 시작되는 샘플에서는 V1이 아직 0이므로 과전압이 I·R0여야 합니다"


def test_long_pulse_reaches_r0_plus_r1():
    eta = mod.overpotential(R0, R1, C1, I_PULSE, DT)
    assert np.isclose(eta[249], 3.0 * (R0 + R1), rtol=1e-2), "τ보다 충분히 긴 펄스 끝에서 과전압은 I·(R0 + R1)에 가까워야 합니다"


def test_relaxation_decays_by_a():
    eta = mod.overpotential(R0, R1, C1, I_PULSE, DT)
    a = np.exp(-DT / (R1 * C1))
    assert np.allclose(eta[261:300] / eta[260:299], a), "휴지 구간에서 V1은 한 step마다 a = exp(-Δt/τ)배로 줄어야 합니다"


def test_residual_zero_at_truth():
    eta = mod.overpotential(R0, R1, C1, I_PULSE, DT)
    r = mod.residual(np.log([R0, R1, C1]), I_PULSE, DT, eta)
    assert np.allclose(r, 0.0, atol=1e-12), "참 파라미터에서 잔차는 0이어야 합니다"


def test_least_squares_recovers_parameters():
    rng = np.random.default_rng(0)
    eta = mod.overpotential(R0, R1, C1, I_PULSE, DT) + rng.normal(0, 1e-3, 400)
    fit = least_squares(mod.residual, np.log([0.01, 0.01, 1000.0]), args=(I_PULSE, DT, eta))
    assert np.allclose(np.exp(fit.x), [R0, R1, C1], rtol=0.02), f"추정값 {np.exp(fit.x)}가 참값과 2% 넘게 다릅니다"
