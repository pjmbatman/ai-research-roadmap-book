"""11.5 퀴즈(content/11-probability/05-expectation-and-limits/quiz.toml) 정답을 계산으로 검증한다."""
import math
import tomllib
from pathlib import Path

import numpy as np
from scipy import stats

QUIZ = Path(__file__).parents[3] / "content/11-probability/05-expectation-and-limits/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_linearity_without_independence():
    x = rng.normal(size=200_000)
    y = x**2 + 0.5 * x                                       # 상관이 있는 종속 변수
    assert np.isclose(np.mean(x + y), np.mean(x) + np.mean(y))
    assert not np.isclose(np.var(x + y), np.var(x) + np.var(y), rtol=0.05)
    assert not np.isclose(np.mean(x * y), np.mean(x) * np.mean(y), atol=0.1)
    assert correct("linearity-without-independence") == [0]


def test_variance_of_difference():
    cov = np.array([[4.0, 3.0], [3.0, 9.0]])
    a = np.array([1.0, -1.0])
    assert a @ cov @ a == Q["variance-of-difference"]["answer"] == 7
    xy = rng.multivariate_normal([0, 0], cov, size=400_000)
    assert abs(np.var(xy[:, 0] - xy[:, 1]) - 7) < 0.1


def test_zero_correlation_dependence():
    x = rng.normal(size=400_000)
    y = x**2
    assert abs(np.cov(x, y)[0, 1]) < 0.02
    assert np.mean(y[np.abs(x) < 1] >= 1) == 0 and np.mean(y >= 1) > 0.3   # 조건부 확률이 바뀐다
    assert correct("zero-correlation-dependence") == [1]


def test_chebyshev_bound():
    k = 15 / 5
    assert abs(1 / k**2 - Q["chebyshev-bound"]["answer"]) <= Q["chebyshev-bound"]["tolerance"]
    for dist in (stats.norm(50, 5), stats.uniform(50 - 5 * math.sqrt(3), 10 * math.sqrt(3)),
                 stats.laplace(50, 5 / math.sqrt(2))):
        assert dist.sf(65) + dist.cdf(35) <= 1 / k**2                       # 경계는 항상 성립


def test_jensen_elbo():
    w = rng.gamma(2.0, 1.5, size=400_000)
    assert np.log(np.mean(w)) > np.mean(np.log(w))
    c = np.full(10, 3.0)                                                   # 상수면 등호
    assert np.isclose(np.log(np.mean(c)), np.mean(np.log(c)))
    assert correct("jensen-elbo") == [0]


def test_hoeffding_sample_size():
    t, delta = 0.05, 0.05
    n = math.ceil(math.log(2 / delta) / (2 * t**2))
    assert 2 * math.exp(-2 * n * t**2) <= delta < 2 * math.exp(-2 * (n - 1) * t**2)
    assert abs(n - Q["hoeffding-sample-size"]["answer"]) <= Q["hoeffding-sample-size"]["tolerance"]
    assert math.isclose(0.25 / (delta * t**2), 2000)


def test_clt_heavy_tail():
    for n in (10, 1000):
        m = rng.standard_cauchy(size=(5000, n)).mean(axis=1)
        assert abs(np.subtract(*np.quantile(m, [0.75, 0.25])) - 2.0) < 0.15  # 표준 Cauchy의 IQR = 2
    m = (rng.pareto(1.5, size=(2000, 2000)) + 1).mean(axis=1)               # 참 평균 3
    assert abs(np.median(m) - 3) < 0.3
    assert correct("clt-heavy-tail") == [0, 1, 2]
