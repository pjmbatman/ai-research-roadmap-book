"""11.2 퀴즈(content/11-probability/02-distributions/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy import integrate, stats

QUIZ = Path(__file__).parents[3] / "content/11-probability/02-distributions/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_photon_count_model():
    # 1 ms를 n칸으로 나눠 칸마다 확률 lam/n으로 광자가 온다 -> n이 커지면 Poisson
    lam, k = 2.0, np.arange(10)
    assert np.abs(stats.binom.pmf(k, 100_000, lam / 100_000) - stats.poisson.pmf(k, lam)).max() < 1e-5
    assert correct("photon-count-model") == [0]


def test_memoryless_lifetime():
    T = stats.expon(scale=2000.0)
    assert np.isclose(T.sf(1500) / T.sf(1000), T.sf(500))
    G = stats.gamma(2.0, scale=1000.0)            # 마모가 있는 수명이면 성립하지 않는다
    assert G.sf(1500) / G.sf(1000) < G.sf(500)
    assert correct("memoryless-lifetime") == [1]


def test_poisson_relative_noise():
    assert np.isclose(stats.poisson(100).std(), 10.0)
    assert abs(Q["poisson-relative-noise"]["answer"] - 10.0) <= Q["poisson-relative-noise"]["tolerance"]


def test_bernoulli_natural_parameter():
    pi = 0.3
    eta = np.log(pi / (1 - pi))
    A = np.log1p(np.exp(eta))
    for x in (0, 1):
        assert np.isclose(np.exp(eta * x - A), pi**x * (1 - pi) ** (1 - x))
    assert correct("bernoulli-natural-parameter") == [2]


def test_log_partition_derivatives():
    # 정규분포(분산 1)의 T(x) = x, eta = mu: A(eta) = eta^2 / 2 -> A' = mu, A'' = 1 = Var
    def A(eta):
        return np.log(integrate.quad(lambda x: np.exp(eta * x - x**2 / 2), -30, 30)[0])
    eta, h = 0.7, 1e-3
    d1 = (A(eta + h) - A(eta - h)) / (2 * h)
    d2 = (A(eta + h) - 2 * A(eta) + A(eta - h)) / h**2
    assert abs(d1 - eta) < 1e-5 and abs(d2 - 1.0) < 1e-3
    assert correct("log-partition-derivatives") == [0]


def test_gaussian_fit_heavy_tail():
    x = stats.t.rvs(df=3, scale=0.2, size=20_000, random_state=rng)
    mu, sd = stats.norm.fit(x)
    p_fit = 2 * stats.norm.sf(1.0, 0, sd)
    assert p_fit < 0.5 * np.mean(np.abs(x) > 1.0)
    assert correct("gaussian-fit-heavy-tail") == [0]


def test_fit_comparison_statements():
    # 모수를 같은 데이터로 추정하면 KS 통계량이 작아진다(p값이 낙관적)
    D_fit, D_true = [], []
    for _ in range(200):
        x = rng.normal(size=50)
        D_fit.append(stats.kstest(x, stats.norm(*stats.norm.fit(x)).cdf).statistic)
        D_true.append(stats.kstest(x, stats.norm.cdf).statistic)
    assert np.mean(D_fit) < 0.8 * np.mean(D_true)
    assert correct("fit-comparison-statements") == [0, 1]
