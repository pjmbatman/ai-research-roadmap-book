"""13.1 퀴즈(content/13-bayesian-inference/01-bayesian-thinking/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy import stats

QUIZ = Path(__file__).parents[3] / "content/13-bayesian-inference/01-bayesian-thinking/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_posterior_predictive_next():
    theta = np.linspace(0, 1, 200001)
    post = stats.binom.pmf(7, 20, theta) * stats.beta.pdf(theta, 2, 2)
    post /= post.sum()
    p_next = np.sum(theta * post)
    assert abs(p_next - Q["posterior-predictive-next"]["answer"]) <= Q["posterior-predictive-next"]["tolerance"]
    assert abs(p_next - 0.35) > 0.02                                       # plug-in MLE와 다르다


def test_evidence_role():
    theta = np.linspace(0, 1, 200001)
    lik = stats.binom.pmf(7, 20, theta)
    evidence = np.mean(lik * stats.beta.pdf(theta, 2, 2))                # [0, 1] 균등 격자 평균 = 적분
    assert evidence < lik.max()                                           # 최댓값이 아니다
    assert correct("evidence-role") == [0]


def test_predictive_wider():
    ks = np.arange(21)
    sd = lambda p: np.sqrt(np.sum(ks**2 * p) - np.sum(ks * p) ** 2)
    assert sd(stats.betabinom.pmf(ks, 20, 9, 15)) > sd(stats.binom.pmf(ks, 20, 0.35))
    # 균등 사전 Beta(1, 1)에서도 더 넓다 (사후 Beta(8, 14), MLE 0.35)
    assert sd(stats.betabinom.pmf(ks, 20, 8, 14)) > sd(stats.binom.pmf(ks, 20, 0.35))
    assert correct("predictive-wider") == [0]


def test_combine_two_days():
    a, b = 2, 2
    for h, n in ((3, 4), (2, 6)):
        a, b = a + h, b + n - h
    assert (a, b) == (7, 7)
    # 각자 사전을 넣은 두 사후 Beta(5,3), Beta(4,6)을 곱하면 지수가 (4+3, 2+5) → Beta(8, 8)
    assert (5 - 1 + 4 - 1 + 1, 3 - 1 + 6 - 1 + 1) == (8, 8)
    assert np.isclose((5 / 8 + 4 / 10) / 2, 0.5125)
    assert correct("combine-two-days") == [0]


def test_credible_vs_confidence():
    n, R = 20, 20000
    ks = np.arange(n + 1)
    cp = (np.nan_to_num(stats.beta.ppf(0.025, ks, n - ks + 1), nan=0.0),
          np.nan_to_num(stats.beta.ppf(0.975, ks + 1, n - ks), nan=1.0))
    cr = stats.beta.ppf([[0.025], [0.975]], 1 + ks, 1 + n - ks)
    k = rng.binomial(n, 0.05, size=R)
    assert np.mean((cp[0][k] <= 0.05) & (0.05 <= cp[1][k])) >= 0.95      # 고정 θ에서 신뢰구간 보장
    assert np.mean((cr[0][k] <= 0.05) & (0.05 <= cr[1][k])) < 0.94       # 신용구간은 보장 없음
    th = rng.beta(1, 1, size=R)
    k = rng.binomial(n, th)
    assert abs(np.mean((cr[0][k] <= th) & (th <= cr[1][k])) - 0.95) < 0.01   # 사전 평균에서는 95%
    assert correct("credible-vs-confidence") == [0, 1]


def test_map_not_bayesian():
    a, b = 5, 2
    assert np.isclose((a - 1) / (a + b - 2), 0.8)
    phi = np.linspace(-6, 8, 140001)
    th = 1 / (1 + np.exp(-phi))
    dens = stats.beta.pdf(th, a, b) * th * (1 - th)
    assert abs(th[np.argmax(dens)] - 5 / 7) < 1e-3
    med_phi = th[np.searchsorted(np.cumsum(dens) / dens.sum(), 0.5)]
    assert abs(med_phi - stats.beta.median(a, b)) < 1e-3
    assert correct("map-not-bayesian") == [0]
