"""25.1 퀴즈(content/25-scientific-discovery/01-simulation-based-inference/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/25-scientific-discovery/01-simulation-based-inference/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def ar1(rho, rng, T=50):
    x = np.zeros((len(rho), T))
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + rng.normal(size=len(rho))
    return x


def abc_std(stat, rho, s_obs, q):
    d = np.abs(stat - s_obs) / stat.std()
    return rho[d <= np.quantile(d, q)].std()


def test_uninformative_summary():
    rng = np.random.default_rng(0)
    rho = rng.uniform(0, 1, 20000)
    x = ar1(rho, rng)
    xc = x - x.mean(1, keepdims=True)
    ac1 = (xc[:, 1:] * xc[:, :-1]).sum(1) / (xc**2).sum(1)
    x_obs = ar1(np.array([0.6]), rng)[0]
    prior_sd = 1 / np.sqrt(12)
    sd_mean = abc_std(x.mean(1), rho, x_obs.mean(), 0.01)
    xo = x_obs - x_obs.mean()
    sd_ac = abc_std(ac1, rho, (xo[1:] * xo[:-1]).sum() / (xo**2).sum(), 0.01)
    assert sd_mean > 0.75 * prior_sd                              # 평균: 사전분포 수준으로 넓다
    assert sd_ac < 0.6 * prior_sd                                 # 자기상관: 좁다
    assert correct("uninformative-summary") == [0]


def test_tolerance_to_prior():
    rng = np.random.default_rng(1)
    rho = rng.uniform(0, 1, 20000)
    stat = rho + 0.1 * rng.normal(size=20000)
    sds = [abc_std(stat, rho, 0.3, q) for q in (0.01, 0.5, 1.0)]
    assert sds[0] < sds[1] < sds[2]
    assert np.isclose(sds[2], rho.std())                          # q = 1: 사전분포 그대로
    assert correct("tolerance-to-prior") == [0]


def test_prior_interval_width():
    lo, hi = 0.05, 0.95                                           # U(0, 1)의 중앙 90% 구간
    rng = np.random.default_rng(2)
    theta = rng.uniform(0, 1, 100000)
    assert abs(np.mean((lo <= theta) & (theta <= hi)) - 0.9) < 0.005   # coverage 0.9
    q = Q["prior-interval-width"]
    assert np.isclose(hi - lo, q["answer"])
    s = q["slider"]
    assert s["min"] <= q["answer"] <= s["max"]
    assert np.isclose(round((q["answer"] - s["min"]) / s["step"]) * s["step"] + s["min"], q["answer"])


def test_low_coverage_binomial_se():
    se = np.sqrt(0.9 * 0.1 / 300)
    assert 0.017 < se < 0.018                                     # 해설의 "약 0.017"
    assert (0.9 - 0.63) / se > 10                                 # 우연으로 설명할 수 없는 차이
    assert correct("low-coverage-meaning") == [0]


def test_choice_keys():
    assert correct("likelihood-free-meaning") == [0]
    assert correct("npe-amortization") == [0, 1]
