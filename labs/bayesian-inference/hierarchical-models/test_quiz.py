"""13.3 퀴즈(content/13-bayesian-inference/03-hierarchical-models/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy import stats

QUIZ = Path(__file__).parents[3] / "content/13-bayesian-inference/03-hierarchical-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_hierarchical_structure():
    assert correct("hierarchical-structure") == [0]


def test_shrinkage_factor():
    sigma, tau, mu, n, ybar = 0.6, 0.3, 2.0, 4, 1.6
    th = np.linspace(0, 4, 400001)                                   # 격자 사후로 직접 확인
    logpost = stats.norm.logpdf(ybar, th, sigma / np.sqrt(n)) + stats.norm.logpdf(th, mu, tau)
    p = np.exp(logpost - logpost.max())
    q = Q["shrinkage-factor"]
    assert abs(np.sum(th * p) / p.sum() - q["answer"]) <= q["tolerance"]


def test_which_shrinks_most():
    sigma, tau = 0.6, 0.3
    n = np.array([1, 5, 40])
    B = (sigma**2 / n) / (sigma**2 / n + tau**2)
    assert np.argmax(B) == 0
    assert correct("which-shrinks-most") == [0]


def test_pooling_extremes():
    se, ybar, mu = np.array([0.6, 0.2]), np.array([1.5, 2.5]), 2.0
    for tau, target in ((1e-6, np.full(2, mu)), (1e6, ybar)):
        B = se**2 / (se**2 + tau**2)
        assert np.allclose(B * mu + (1 - B) * ybar, target, atol=1e-6)
    # 그룹 8개에서 empirical Bayes의 τ가 0 근처로 붕괴하는 경우가 있다
    n_j = np.array([1, 2, 3, 5, 8, 12, 20, 40])
    se = 0.6 / np.sqrt(n_j)
    taus = np.linspace(0.005, 1.5, 300)
    collapsed = 0
    for _ in range(500):
        yb = 2.0 + 0.3 * rng.normal(size=8) + se * rng.normal(size=8)
        v = taus[:, None] ** 2 + se**2
        m = np.sum(yb / v, 1) / np.sum(1 / v, 1)
        ll = -0.5 * np.sum(np.log(v) + (yb - m[:, None]) ** 2 / v, 1)
        collapsed += taus[np.argmax(ll)] < 0.01
    assert collapsed / 500 > 0.05
    assert correct("pooling-extremes") == [0, 1]


def test_funnel_cause():
    # 깔때기: log τ가 작을수록 θ의 조건부 표준편차(τ)가 지수적으로 작아진다
    assert np.exp(-3) / np.exp(1) < 0.02
    assert correct("funnel-cause") == [0]


def test_noncentered_form():
    eta = rng.normal(size=200_000)
    mu, tau = 1.0, 0.25
    assert abs(np.std(mu + tau * eta) - tau) < 0.005
    assert abs(np.std(mu + eta / tau) - 1 / tau) < 0.05                # 다른 모델
    assert correct("noncentered-form") == [0]
