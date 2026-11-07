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


cycle_life = load().cycle_life


def step_record():
    cycle = np.arange(1, 201)
    q = np.where(cycle <= 100, 3.0, 2.0)          # 사이클 101부터 2.0 Ah
    return cycle, q


def test_step_with_trailing_median():
    cycle, q = step_record()
    life, observed = cycle_life(cycle, q, ref_ah=3.0, frac=0.8, window=5)
    assert observed, "기준 아래로 떨어졌으니 observed는 True여야 합니다"
    assert life == 103, "창 5의 후행 중앙값은 낮은 값이 3개 쌓인 사이클 103에서 처음 2.4 아래가 됩니다"


def test_single_spike_does_not_trigger():
    cycle = np.arange(1, 101)
    q = np.full(100, 2.9)
    q[40] = 1.0                                     # 한 사이클만 측정이 튐
    life, observed = cycle_life(cycle, q, ref_ah=3.0)
    assert not observed and life == 100, "한 사이클의 튐으로 수명이 정해지면 안 됩니다(중앙값 평활)"


def test_censored():
    cycle = np.arange(1, 501)
    q = 3.0 - 1e-3 * cycle                          # 500 사이클에 2.5 Ah: 2.4에 닿지 않음
    assert cycle_life(cycle, q, ref_ah=3.0) == (500, False), "끝까지 기준에 닿지 않으면 (마지막 사이클, False)"


def test_nan_and_missing_cycles():
    cycle = np.r_[np.arange(1, 51), np.arange(61, 111)]   # 사이클 51–60 기록 없음
    q = 3.0 - 0.01 * cycle
    q[[5, 6, 70]] = np.nan
    life, observed = cycle_life(cycle, q, ref_ah=3.0, window=3)
    # 2.4 아래가 되는 첫 사이클은 61(2.39). 후행 창 3에서 두 번째 값이 쌓인 사이클 62에서 중앙값 < 2.4
    assert observed and life == 62, "NaN은 빼고, 반환은 위치가 아니라 cycle 번호여야 합니다"


def test_reference_definition_matters():
    cycle = np.arange(1, 1501)
    q = 3.08 - 6e-4 * cycle
    life_nom, _ = cycle_life(cycle, q, ref_ah=3.0, window=1)    # 공칭 기준: 2.4 Ah
    life_ini, _ = cycle_life(cycle, q, ref_ah=3.08, window=1)   # 첫 측정 기준: 2.464 Ah
    assert life_nom == 1134, "공칭 3.0 Ah 기준이면 2.4 Ah 아래가 되는 사이클 1134"
    assert life_ini == 1027, "첫 측정 3.08 Ah 기준이면 2.464 Ah 아래가 되는 사이클 1027"
