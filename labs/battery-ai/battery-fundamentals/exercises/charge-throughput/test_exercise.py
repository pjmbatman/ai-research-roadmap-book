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


throughput = load().throughput


def cycle_record():
    """1 s 간격: 2C(6 A) 방전 1800 s → 휴지 600 s → 1C(3 A) 충전 3600 s. 방전 + 규약."""
    i_a = np.r_[np.full(1800, 6.0), np.zeros(600), np.full(3600, -3.0), 0.0]   # (T,)
    t_s = np.arange(len(i_a), dtype=float)
    v_v = np.where(i_a > 0, 3.6, np.where(i_a < 0, 4.0, 3.8))
    return t_s, i_a, v_v


def test_constant_current_capacity():
    t_s, i_a, v_v = cycle_record()
    r = throughput(t_s, i_a, v_v, q_nom_ah=3.0)
    assert np.isclose(r["discharge_ah"], 3.0), "6 A x 1800 s = 3.0 Ah여야 합니다(초를 시간으로 바꿨는지 확인)"
    assert np.isclose(r["charge_ah"], 3.0), "3 A x 3600 s = 3.0 Ah여야 합니다"
    assert r["discharge_ah"] >= 0 and r["charge_ah"] >= 0, "전하량은 0 이상으로 반환합니다"


def test_discharge_energy_uses_voltage():
    t_s, i_a, v_v = cycle_record()
    r = throughput(t_s, i_a, v_v, q_nom_ah=3.0)
    assert np.isclose(r["discharge_wh"], 3.0 * 3.6), "방전 에너지는 방전 구간의 V·I만 적분합니다(3.0 Ah x 3.6 V)"


def test_efc():
    t_s, i_a, v_v = cycle_record()
    r = throughput(t_s, i_a, v_v, q_nom_ah=2.0)
    assert np.isclose(r["efc"], 1.5), "EFC = 방전 전하량 / 공칭 용량 = 3.0 / 2.0"


def test_sign_convention():
    t_s, i_a, v_v = cycle_record()
    a = throughput(t_s, i_a, v_v, 3.0, discharge_sign=+1)
    b = throughput(t_s, -i_a, v_v, 3.0, discharge_sign=-1)
    for k in ["discharge_ah", "charge_ah", "discharge_wh", "efc"]:
        assert np.isclose(a[k], b[k]), f"부호 규약만 다른 같은 기록이면 {k}가 같아야 합니다"


def test_irregular_sampling_zero_order_hold():
    t_s = np.array([0.0, 10.0, 70.0, 100.0])
    i_a = np.array([3.6, 1.8, -7.2, 99.0])     # 마지막 측정은 기여하지 않는다
    v_v = np.array([4.0, 3.9, 4.1, 3.0])
    r = throughput(t_s, i_a, v_v, 3.0)
    assert np.isclose(r["discharge_ah"], (3.6 * 10 + 1.8 * 60) / 3600), "구간 시작 값 x 구간 길이(영차 유지)로 적분합니다"
    assert np.isclose(r["charge_ah"], 7.2 * 30 / 3600)
    assert np.isclose(r["discharge_wh"], (3.6 * 4.0 * 10 + 1.8 * 3.9 * 60) / 3600)


@pytest.mark.parametrize("n", [1, 5])
def test_partial_cycles(n):
    # SOC 30% ↔ 80% 부분 사이클 n회: 사이클마다 1.5 Ah 방전, 1.5 Ah 충전 (3 A)
    one = np.r_[np.full(1800, 3.0), np.full(1800, -3.0)]
    i_a = np.r_[np.tile(one, n), 0.0]
    t_s = np.arange(len(i_a), dtype=float)
    r = throughput(t_s, i_a, np.full(len(i_a), 3.7), 3.0)
    assert np.isclose(r["efc"], 0.5 * n), "부분 사이클 n회의 EFC는 0.5n입니다"
