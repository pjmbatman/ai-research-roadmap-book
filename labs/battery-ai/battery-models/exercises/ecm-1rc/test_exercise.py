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


simulate_1rc = load().simulate_1rc

R0, R1, C1, Q = 0.020, 0.015, 2000.0, 3.0      # tau = 30 s
TAU = R1 * C1


def flat_ocv(soc):
    return 3.7 + 0.0 * np.asarray(soc)          # SOC와 무관한 OCV: 과전압만 보인다


def pulse(n_rest=10, n_on=600, n_off=600, i=3.0):
    return np.r_[np.zeros(n_rest), np.full(n_on, i), np.zeros(n_off)]


def test_instant_drop_is_r0():
    v, _ = simulate_1rc(pulse(), 1.0, R0, R1, C1, Q, 0.8, flat_ocv)
    assert np.isclose(v[9] - v[10], 3.0 * R0), "전류가 바뀌는 순간의 전압 변화는 ΔI·R0여야 합니다"


def test_steady_state_and_time_constant():
    v, _ = simulate_1rc(pulse(), 1.0, R0, R1, C1, Q, 0.8, flat_ocv)
    assert np.isclose(3.7 - v[609], 3.0 * (R0 + R1), atol=1e-6), "충분히 긴 펄스 끝의 과전압은 I(R0 + R1)이어야 합니다"
    over = 3.7 - v[10:610] - 3.0 * R0               # RC 몫의 과전압
    k63 = np.argmax(over >= 0.632 * 3.0 * R1)
    assert abs(k63 - TAU) <= 1, "RC 몫의 과전압이 63%에 이르는 시간이 tau = R1 C1이어야 합니다"


def test_soc_coulomb_count():
    i = pulse()
    _, soc = simulate_1rc(i, 1.0, R0, R1, C1, Q, 0.8, flat_ocv)
    assert soc[0] == 0.8, "soc[0]은 soc0여야 합니다"
    assert np.isclose(soc[-1], 0.8 - np.sum(i[:-1]) / 3600 / Q), "SOC는 영차 유지로 적분한 전하량만큼 줄어야 합니다"


def test_uses_ocv_of_soc():
    ocv = lambda s: 3.4 + 0.6 * np.asarray(s)
    i = np.full(1801, 3.0)
    v, soc = simulate_1rc(i, 1.0, R0, R1, C1, Q, 0.9, ocv)
    assert np.isclose(soc[-1], 0.9 - 0.5), "3 A로 1800 s면 1.5 Ah, 즉 SOC 0.5만큼 줄어야 합니다"
    assert np.isclose(v[-1], ocv(soc[-1]) - 3.0 * (R0 + R1), atol=1e-4), "전압은 현재 SOC의 OCV에서 과전압을 뺀 값입니다"


def test_large_step_is_exact_and_stable():
    i_fine = pulse(n_rest=90, n_on=360, n_off=450)          # 1 s 간격, 900 s
    v_fine, _ = simulate_1rc(i_fine, 1.0, R0, R1, C1, Q, 0.8, flat_ocv)
    i_coarse = i_fine[::90]                                  # dt = 90 s = 3 tau
    v_coarse, _ = simulate_1rc(i_coarse, 90.0, R0, R1, C1, Q, 0.8, flat_ocv)
    assert np.all(np.isfinite(v_coarse)) and np.max(np.abs(v_coarse)) < 10, "dt > 2 tau에서도 발산하면 안 됩니다(전진 Euler를 쓰지 않았는지 확인)"
    assert np.allclose(v_coarse, v_fine[::90], atol=1e-9), "영차 유지의 정확한 해는 step 크기와 무관하게 같은 시각에서 같은 값을 줘야 합니다"
