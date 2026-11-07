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


m = load()


def test_single_sensor_is_bayes():
    prior, s, f = 0.01, 0.9, 0.05
    expected = s * prior / (s * prior + f * (1 - prior))
    assert m.posterior_fault(prior, [s], [f], [1]) == pytest.approx(expected)
    quiet = (1 - s) * prior / ((1 - s) * prior + (1 - f) * (1 - prior))
    assert m.posterior_fault(prior, [s], [f], [0]) == pytest.approx(quiet), "경보가 없는 것도 정보입니다"


def test_sequential_equals_batch():
    sens, fpr, alarms = [0.9, 0.7, 0.95], [0.05, 0.2, 0.01], [1, 0, 1]
    p = 0.02
    for s, f, a in zip(sens, fpr, alarms):
        p = m.posterior_fault(p, [s], [f], [a])       # 오늘의 사후 = 다음 단계의 사전
    assert m.posterior_fault(0.02, sens, fpr, alarms) == pytest.approx(p)


def test_matches_simulation():
    rng = np.random.default_rng(0)
    n, prior = 400_000, 0.1
    sens, fpr = np.array([0.9, 0.8]), np.array([0.05, 0.1])
    fault = rng.random(n) < prior
    alarms = rng.random((n, 2)) < np.where(fault[:, None], sens, fpr)   # (n, 2)
    for pattern in ([1, 1], [1, 0], [0, 0]):
        sel = np.all(alarms == np.array(pattern, dtype=bool), axis=1)
        assert m.posterior_fault(prior, sens, fpr, pattern) == pytest.approx(fault[sel].mean(), abs=0.01)


def test_many_sensors_stable():
    k = 2000
    sens, fpr = np.full(k, 0.6), np.full(k, 0.4)
    alarms = np.r_[np.ones(1000, int), np.zeros(1000, int)]
    p = m.posterior_fault(0.3, sens, fpr, alarms)              # 증거가 정확히 상쇄된다
    assert np.isfinite(p), "우도를 곱하면 underflow합니다. 로그로 더하세요"
    assert p == pytest.approx(0.3)
    assert m.posterior_fault(0.3, sens, fpr, np.ones(k, int)) == pytest.approx(1.0)
    assert m.posterior_fault(0.3, sens, fpr, np.zeros(k, int)) == pytest.approx(0.0, abs=1e-12)
