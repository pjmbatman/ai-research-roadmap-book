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
counts = np.array([3, 12, 0, 41])
exposure_s = np.array([1.0, 5.0, 0.5, 20.0])


def test_posterior_parameters():
    a, b = m.posterior(2.0, 1.0, counts, exposure_s)
    assert (a, b) == pytest.approx((58.0, 27.5)), "shape에는 계수의 합을, rate에는 노출 시간의 합을 더합니다"


def test_posterior_matches_grid():
    a, b = m.posterior(2.0, 1.0, counts, exposure_s)
    rate = np.linspace(1e-4, 6, 20001)
    logpost = stats.gamma.logpdf(rate, 2.0, scale=1.0) \
        + stats.poisson.logpmf(counts[:, None], rate * exposure_s[:, None]).sum(axis=0)
    grid = np.exp(logpost - logpost.max())
    grid /= grid.sum() * (rate[1] - rate[0])
    assert np.max(np.abs(grid - stats.gamma.pdf(rate, a, scale=1 / b))) < 1e-4, "격자에서 계산한 사후분포와 다릅니다"


def test_exposure_matters():
    # 같은 계수라도 노출 시간이 길면 계수율이 낮다
    _, b_short = m.posterior(1.0, 0.1, [10], [1.0])
    a_long, b_long = m.posterior(1.0, 0.1, [10], [10.0])
    assert a_long / b_long < 1.2, "노출 시간 10 s 동안 10개면 계수율은 약 1 /s입니다"
    assert b_long > b_short, "노출 시간은 rate 파라미터에 더해집니다"


def test_sequential_equals_batch():
    a, b = 2.0, 1.0
    for c, t in zip(counts, exposure_s):
        a, b = m.posterior(a, b, [c], [t])
    assert (a, b) == pytest.approx(m.posterior(2.0, 1.0, counts, exposure_s)), "하나씩 갱신한 결과가 한꺼번에 갱신한 결과와 다릅니다"


def test_credible_interval():
    lo, hi = m.credible_interval(58.0, 27.5)
    assert (lo, hi) == pytest.approx((1.6017, 2.6853), abs=1e-3)
    mass = stats.gamma.cdf(hi, 58.0, scale=1 / 27.5) - stats.gamma.cdf(lo, 58.0, scale=1 / 27.5)
    assert mass == pytest.approx(0.95), "구간 안의 사후확률이 level과 같아야 합니다"
    lo90, hi90 = m.credible_interval(58.0, 27.5, level=0.9)
    assert lo < lo90 < hi90 < hi, "90% 구간은 95% 구간보다 좁아야 합니다"


def test_prob_rate_above():
    p = m.prob_rate_above(58.0, 27.5, 2.5)
    assert p == pytest.approx(stats.gamma.sf(2.5, 58.0, scale=1 / 27.5))
    rng = np.random.default_rng(0)
    draws = rng.gamma(58.0, 1 / 27.5, size=200_000)
    assert p == pytest.approx(np.mean(draws > 2.5), abs=0.005), "샘플링으로 확인한 꼬리 확률과 다릅니다(scale = 1/b)"
