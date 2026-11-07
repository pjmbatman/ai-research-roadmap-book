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


summarize_steps = load().summarize_steps


def make(data, comp, bs=64):
    return [{"step": i, "batch_size": bs, "data_s": d, "compute_s": c} for i, (d, c) in enumerate(zip(data, comp))]


STEPS = make([0.90, 0.04, 0.04, 0.04, 0.04], [2.10, 0.06, 0.06, 0.06, 0.06])


def test_warmup_is_excluded():
    r = summarize_steps(STEPS, warmup=1)
    assert np.isclose(r["throughput"], 4 * 64 / (4 * 0.10)), "처리량은 warm-up step을 뺀 샘플 수 / 시간입니다"
    with_warmup = 5 * 64 / (3.0 + 4 * 0.10)
    assert not np.isclose(r["throughput"], with_warmup), "첫 step의 초기화 비용이 처리량에 섞였습니다"


def test_data_fraction():
    r = summarize_steps(STEPS, warmup=1)
    assert np.isclose(r["data_fraction"], 0.4), "데이터 대기 비율 = data_s 합 / (data_s + compute_s) 합"


def test_overlap_bound_when_compute_dominates():
    r = summarize_steps(STEPS, warmup=1)
    assert np.isclose(r["overlap_step_s"], 0.06), "완전히 겹치면 step 시간은 max(data_s, compute_s)"
    assert np.isclose(r["overlap_speedup"], 0.10 / 0.06)


def test_overlap_bound_when_data_dominates():
    steps = make([0.0, 0.30, 0.30, 0.30], [0.0, 0.10, 0.10, 0.10], bs=32)
    r = summarize_steps(steps, warmup=1)
    assert np.isclose(r["data_fraction"], 0.75)
    assert np.isclose(r["overlap_step_s"], 0.30), "데이터가 더 느리면 겹쳐도 데이터 시간 아래로 내려가지 않습니다"
    assert np.isclose(r["overlap_speedup"], 0.40 / 0.30)


def test_mixed_steps_use_per_step_max():
    steps = make([0.05, 0.02], [0.03, 0.08])
    r = summarize_steps(steps, warmup=0)
    assert np.isclose(r["overlap_step_s"], (0.05 + 0.08) / 2), "max는 step마다 따로 계산한 뒤 평균합니다"


def test_warmup_too_large():
    with pytest.raises(ValueError):
        summarize_steps(STEPS, warmup=5)
