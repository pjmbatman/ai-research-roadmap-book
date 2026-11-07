"""13.8 퀴즈(content/13-bayesian-inference/08-bayesian-workflow/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

QUIZ = Path(__file__).parents[3] / "content/13-bayesian-inference/08-bayesian-workflow/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_vague_prior_predictive():
    lam = np.exp(np.random.default_rng(0).normal(0, 10, 100000))
    assert np.mean(lam < 1) > 0.45 and np.mean(lam > 1000) > 0.2
    assert correct("vague-prior-predictive") == [0]


def test_ppc_overdispersion():
    rng = np.random.default_rng(1)
    y = rng.negative_binomial(5, 5 / 35, size=40)
    reps = rng.poisson(y.mean(), size=(4000, 40))
    disp = reps.var(axis=1) / reps.mean(axis=1)
    assert np.mean(disp >= y.var() / y.mean()) < 0.001            # 과분산을 재현하지 못한다
    assert 0.2 < np.mean(reps.mean(axis=1) >= y.mean()) < 0.8     # 평균은 통과한다
    assert correct("ppc-overdispersion") == [0]


def sbc_counts(var_scale, rng, n_sims=2000, L=99):
    counts = np.zeros(10, dtype=int)
    for _ in range(n_sims):
        lam = rng.gamma(2.0, 10.0)
        y = rng.poisson(lam, size=20)
        shape = (2.0 + y.sum()) / var_scale
        draws = rng.gamma(shape, (2.0 + y.sum()) / (0.1 + 20) / shape, size=L)
        counts[np.sum(draws < lam) // 10] += 1
    return counts


def test_sbc_shapes():
    rng = np.random.default_rng(2)
    narrow, wide = sbc_counts(0.25, rng), sbc_counts(4.0, rng)
    assert narrow[0] + narrow[-1] > 3 * narrow[4:6].sum()          # 좁은 사후: U자
    assert wide[4:6].sum() > 3 * (wide[0] + wide[-1])              # 넓은 사후: ∩자
    assert correct("sbc-u-shape") == [0]


def test_sbc_rank_bin():
    assert 1000 * 10 / 100 == Q["sbc-rank-bin"]["answer"]


def test_elpd_difference():
    assert abs((-150.8 - -179.8) - Q["elpd-difference"]["answer"]) < 1e-9


def test_bayes_factor_prior_width():
    # 모델 0: theta = 0, 모델 1: theta ~ N(0, s^2), y ~ N(theta, 1). 사전 폭 s만 바꾼다.
    y = 1.5
    log_bf = lambda s: norm.logpdf(y, 0, np.sqrt(1 + s**2)) - norm.logpdf(y, 0, 1)
    assert log_bf(1) > 0 and log_bf(100) < -2                      # 사전을 넓히자 선호가 뒤집힌다
    # 같은 상황에서 s = 10과 s = 100의 사후분포는 거의 같다
    post_sd = lambda s: np.sqrt(1 / (1 + 1 / s**2))
    assert abs(post_sd(10) - post_sd(100)) < 0.01
    assert correct("comparison-statements") == [0, 1]


def test_loo_identity_on_grid():
    # 격자 사후에서 -log E_post[1 / p(y_i | th)]가 y_i를 뺀 사후의 예측밀도와 같은지 확인
    rng = np.random.default_rng(3)
    y = rng.normal(1.0, 1.0, size=6)
    th = np.linspace(-6, 8, 4001)
    ll = norm.logpdf(y[:, None], th[None, :], 1.0)                # (6, G)
    log_prior = norm.logpdf(th, 0, 3)
    lw = ll.sum(0) + log_prior
    lw -= logsumexp(lw)
    loo_identity = -logsumexp(-ll[0] + lw)
    lw_minus = ll[1:].sum(0) + log_prior
    lw_minus -= logsumexp(lw_minus)
    loo_direct = logsumexp(ll[0] + lw_minus)
    assert abs(loo_identity - loo_direct) < 1e-8
