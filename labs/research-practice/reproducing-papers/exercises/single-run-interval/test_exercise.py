import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()
RUNS = [0.850, 0.861, 0.842, 0.857, 0.848, 0.853, 0.839, 0.862, 0.846, 0.852]


def test_mean_interval_values():
    x = np.array(RUNS)
    half = stats.t.ppf(0.975, 9) * x.std(ddof=1) / np.sqrt(10)
    assert m.mean_interval(RUNS) == pytest.approx((x.mean() - half, x.mean() + half))


def test_single_run_interval_values():
    x = np.array(RUNS)
    half = stats.t.ppf(0.975, 9) * x.std(ddof=1) * np.sqrt(1.1)
    assert m.single_run_interval(RUNS) == pytest.approx((x.mean() - half, x.mean() + half))


def test_prediction_wider_than_mean():
    lo_m, hi_m = m.mean_interval(RUNS)
    lo_p, hi_p = m.single_run_interval(RUNS)
    assert (hi_p - lo_p) / (hi_m - lo_m) == pytest.approx(np.sqrt(11), rel=1e-9), "폭의 비는 sqrt(n + 1)입니다"


def test_is_consistent():
    assert m.is_consistent(0.866, RUNS) is True, "평균 구간 밖이어도 단일 실행으로는 흔한 값입니다"
    lo_m, hi_m = m.mean_interval(RUNS)
    assert not (lo_m <= 0.866 <= hi_m)
    assert m.is_consistent(0.90, RUNS) is False


def test_coverage_simulation():
    rng = np.random.default_rng(0)
    hit_p = hit_m = 0
    trials = 2000
    for _ in range(trials):
        runs = rng.normal(0.85, 0.01, size=5)
        new = rng.normal(0.85, 0.01)                   # 원 논문의 단일 실행
        lo, hi = m.single_run_interval(runs)
        hit_p += lo <= new <= hi
        lo, hi = m.mean_interval(runs)
        hit_m += lo <= new <= hi
    assert hit_p / trials == pytest.approx(0.95, abs=0.015), "예측구간은 새 단일 실행값을 약 95% 덮어야 합니다"
    assert hit_m / trials < 0.75, "평균의 신뢰구간은 단일 실행값을 훨씬 적게 덮습니다"
