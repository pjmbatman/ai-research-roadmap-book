import importlib.util
import os
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


self_normalized_is = load().self_normalized_is


def test_equal_weights():
    fx = np.arange(10.0)
    est, ess = self_normalized_is(np.full(10, -3.0), fx)
    assert np.isclose(ess, 10), f"가중치가 모두 같으면 ESS = N = 10입니다. 현재 {ess}"
    assert np.isclose(est, fx.mean()), "가중치가 모두 같으면 추정값은 단순 평균입니다"


def test_dominant_weight():
    logw = np.zeros(100)
    logw[7] = 20.0                                   # 하나가 e^20배 크다
    est, ess = self_normalized_is(logw, np.arange(100.0))
    assert ess < 1.01, f"가중치 하나가 압도하면 ESS는 1에 가깝습니다. 현재 {ess:.3f}"
    assert np.isclose(est, 7.0, atol=1e-3)


def test_known_ess():
    est, ess = self_normalized_is(np.log([1.0, 1.0, 2.0]), np.array([0.0, 0.0, 3.0]))
    assert np.isclose(ess, 16 / 6), f"가중치 (1, 1, 2)의 ESS는 16/6입니다. 현재 {ess}"
    assert np.isclose(est, 1.5)


def test_large_log_weights():
    rng = np.random.default_rng(0)
    logw = 1000.0 + rng.normal(size=50)
    est, ess = self_normalized_is(logw, np.ones(50))
    assert np.isfinite(est) and np.isfinite(ess), "로그 가중치가 커도 결과가 유한해야 합니다(overflow)"
    assert np.isclose(est, 1.0)


def test_gaussian_second_moment():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2.0, size=20_000)              # 제안 N(0, 2^2)
    logw = -0.5 * z**2 - stats.norm.logpdf(z, 0, 2.0)   # 정규화 상수를 뺀 N(0,1) 로그밀도
    est, ess = self_normalized_is(logw, z**2)
    assert abs(est - 1.0) < 0.05, f"E[X^2] = 1을 추정해야 합니다. 현재 {est:.3f}"
    assert 0.5 * 20_000 < ess < 0.8 * 20_000, f"이 제안분포의 ESS/N은 약 0.66입니다. 현재 {ess / 20_000:.3f}"
