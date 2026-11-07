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
ybar = np.array([1.6, 2.4, 1.9, 2.7])
se = np.array([0.3, 0.6, 0.1, 0.05])


def test_known_value():
    mean, sd = m.partial_pool([1.6], [0.3], 2.0, 0.3)
    assert np.asarray(mean) == pytest.approx([1.8]), "B = 0.09/(0.09+0.09) = 0.5이므로 1.6과 2.0의 중간입니다"
    assert np.asarray(sd) == pytest.approx([np.sqrt(0.045)])


def test_limits():
    mean0, _ = m.partial_pool(ybar, se, 2.0, 1e-6)
    mean_inf, _ = m.partial_pool(ybar, se, 2.0, 1e6)
    assert np.asarray(mean0) == pytest.approx(np.full(4, 2.0), abs=1e-6), "τ → 0이면 complete pooling(모두 μ)입니다"
    assert np.asarray(mean_inf) == pytest.approx(ybar, abs=1e-6), "τ → ∞이면 no pooling(표본평균 그대로)입니다"


def test_noisier_groups_shrink_more():
    yb = np.array([1.0, 1.0, 1.0])
    s = np.array([0.1, 0.3, 1.0])
    mean, sd = m.partial_pool(yb, s, 2.0, 0.3)
    moved = np.asarray(mean) - yb
    assert moved[0] < moved[1] < moved[2], "표준오차가 큰 그룹일수록 μ 쪽으로 더 많이 끌려가야 합니다"
    assert np.all(np.asarray(sd) < s), "사후 표준편차는 표준오차보다 작아야 합니다(모집단 정보가 더해짐)"


def test_marginal_loglik():
    ll = m.marginal_loglik(ybar, se, 2.1, 0.4)
    expected = stats.norm.logpdf(ybar, 2.1, np.sqrt(0.4**2 + se**2)).sum()
    assert ll == pytest.approx(expected), "주변분산은 τ² + se_j²입니다(표준편차가 아니라 분산을 더합니다)"


def test_eb_recovers_tau():
    rng = np.random.default_rng(0)
    J = 400
    s = rng.uniform(0.1, 0.5, J)
    theta = 3.0 + 0.4 * rng.normal(size=J)
    yb = theta + s * rng.normal(size=J)
    mu_hat, tau_hat = m.eb_estimate(yb, s, np.linspace(0.01, 1.5, 150))
    assert mu_hat == pytest.approx(3.0, abs=0.08)
    assert tau_hat == pytest.approx(0.4, abs=0.08), "그룹이 많으면 주변우도의 최대점이 참 τ 근처여야 합니다"


def test_eb_collapses_when_no_spread():
    s = np.full(6, 0.5)
    yb = np.array([2.0, 2.05, 1.95, 2.02, 1.98, 2.0])          # 표준오차보다 훨씬 덜 흩어짐
    _, tau_hat = m.eb_estimate(yb, s, np.linspace(0.01, 1.5, 150))
    assert tau_hat == pytest.approx(0.01), "그룹 평균이 잡음으로 다 설명되면 주변우도는 τ의 최솟값에서 최대입니다"
