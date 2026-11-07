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
rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(-2.0, 0.5, 150), rng.normal(1.5, 1.0, 350)])
pi0, mu0, sd0 = np.array([0.5, 0.5]), np.array([-1.0, 0.0]), np.array([1.0, 1.0])


def test_responsibilities_and_loglik():
    resp, ll = m.e_step(x, pi0, mu0, sd0)
    assert np.shape(resp) == (500, 2), "책임도의 shape은 (N, K)입니다"
    assert np.sum(resp, axis=1) == pytest.approx(np.ones(500)), "각 점의 책임도는 성분에 대해 합이 1입니다"
    dens = pi0 * stats.norm.pdf(x[:, None], mu0, sd0)
    assert ll == pytest.approx(np.log(dens.sum(axis=1)).sum()), "주변 로그우도는 Σ_n log Σ_k π_k N(x_n | μ_k, σ_k²)입니다"
    assert resp == pytest.approx(dens / dens.sum(axis=1, keepdims=True))


def test_mixing_weight_matters():
    resp, _ = m.e_step(np.array([1.0]), np.array([0.2, 0.8]), np.array([0.0, 2.0]), np.array([1.0, 1.0]))
    assert np.asarray(resp)[0] == pytest.approx([0.2, 0.8]), "두 성분의 우도가 같으면 책임도는 혼합 가중치와 같습니다"


def test_far_point_is_stable():
    resp, ll = m.e_step(np.array([0.0, 1000.0]), pi0, mu0, sd0)
    assert np.all(np.isfinite(resp)) and np.isfinite(ll), "먼 점에서 NaN·inf가 납니다. logsumexp로 정규화해 보세요"
    assert np.asarray(resp)[1] == pytest.approx([0.0, 1.0], abs=1e-12), "x = 1000은 평균이 더 큰 성분에 속해야 합니다"


def test_m_step_hard_assignment_is_group_mle():
    z = (x > 0).astype(int)
    resp = np.eye(2)[z]
    pi, mu, sd = m.m_step(x, resp)
    for k in range(2):
        g = x[z == k]
        assert pi[k] == pytest.approx(len(g) / len(x))
        assert mu[k] == pytest.approx(g.mean())
        assert sd[k] == pytest.approx(g.std()), "표준편차는 N_k로 나눈 MLE입니다(ddof=0)"


def test_m_step_weighted_mean():
    pi, mu, sd = m.m_step(np.array([0.0, 2.0, 4.0]), np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]))
    assert mu == pytest.approx([2 / 3, 10 / 3]), "평균은 책임도로 가중한 평균입니다"
    assert pi == pytest.approx([0.5, 0.5])


def test_em_monotone_and_recovers():
    pi, mu, sd = pi0, mu0, sd0
    lls = []
    for _ in range(200):
        resp, ll = m.e_step(x, pi, mu, sd)
        lls.append(ll)
        pi, mu, sd = m.m_step(x, resp)
    assert np.all(np.diff(lls) > -1e-8), "EM의 로그우도는 반복마다 감소하지 않아야 합니다"
    order = np.argsort(mu)
    assert np.asarray(mu)[order] == pytest.approx([-2.0, 1.5], abs=0.15)
    assert np.asarray(pi)[order] == pytest.approx([0.3, 0.7], abs=0.05)
