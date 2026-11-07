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
soc_coulomb, offset_drift_pct = mod.soc_coulomb, mod.offset_drift_pct


def test_constant_discharge():
    t = np.arange(0, 3601.0)
    soc = soc_coulomb(t, np.full(t.size, 1.5), q_ah=3.0, soc0=0.9)
    assert soc.shape == t.shape, "SOC 배열의 길이는 시각 배열과 같아야 합니다"
    assert soc[0] == 0.9, "soc[0]은 soc0여야 합니다"
    assert np.isclose(soc[-1], 0.4), "1.5 A x 1 h = 1.5 Ah, 3 Ah 셀이면 SOC 0.5 감소"


def test_irregular_sampling():
    t = np.array([0.0, 100.0, 400.0, 1000.0])
    i = np.array([3.6, -1.8, 7.2, 50.0])                 # 마지막 측정은 기여하지 않음
    soc = soc_coulomb(t, i, q_ah=2.0, soc0=0.5)
    expected = 0.5 - np.cumsum([3.6 * 100, -1.8 * 300, 7.2 * 600]) / 3600 / 2.0
    assert np.allclose(soc[1:], expected), "구간 시작 전류 x 실제 시간 간격으로 적분해야 합니다"


def test_charge_efficiency_only_on_charge():
    t = np.arange(0, 7201.0)
    i = np.r_[np.full(3600, 2.0), np.full(3600, -2.0), 0.0]   # 2 Ah 방전 후 2 Ah 충전
    soc = soc_coulomb(t, i, q_ah=4.0, soc0=0.8, eta_chg=0.99)
    assert np.isclose(soc[3600], 0.3), "방전에는 효율을 곱하지 않습니다"
    assert np.isclose(soc[-1], 0.3 + 0.99 * 0.5), "충전 전하의 99%만 저장됩니다"


def test_not_clipped():
    t = np.arange(0, 7201.0)
    soc = soc_coulomb(t, np.full(t.size, 3.0), q_ah=3.0, soc0=0.5)
    assert soc[-1] < 0, "SOC를 0–1로 자르지 않아야 오차 누적이 드러납니다"


def test_offset_drift_matches_simulation():
    t = np.arange(0, 10 * 3600 + 1.0, 10.0)
    rng = np.random.default_rng(0)
    i = rng.uniform(-3, 3, t.size)
    err = soc_coulomb(t, i + 0.05, 2.5, 0.5) - soc_coulomb(t, i, 2.5, 0.5)
    assert np.isclose(abs(err[-1]) * 100, offset_drift_pct(0.05, 10, 2.5)), "offset 0.05 A x 10 h = 0.5 Ah → 2.5 Ah의 20%p"
    assert np.isclose(offset_drift_pct(0.05, 10, 2.5), 20.0)
    assert np.isclose(offset_drift_pct(-0.05, 5, 2.5), 10.0), "오차는 시간에 비례하고, 크기로 반환합니다"
