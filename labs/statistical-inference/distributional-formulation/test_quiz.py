"""12.3 퀴즈(content/12-statistical-inference/03-distributional-formulation/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/12-statistical-inference/03-distributional-formulation/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)
y = rng.gamma(2.0, size=20000)                 # 비대칭 분포: 평균·중앙값·최빈값이 모두 다르다
grid = np.linspace(0, 8, 1601)


def argmin(loss):
    return grid[np.argmin([loss(q) for q in grid])]


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_squared_loss_gives_mean():
    assert abs(argmin(lambda q: np.mean((y - q) ** 2)) - y.mean()) < 0.01
    assert correct("squared-loss-optimum") == [0]


def test_absolute_loss_gives_median():
    q = argmin(lambda q: np.mean(np.abs(y - q)))
    assert abs(q - np.median(y)) < 0.01 and abs(q - y.mean()) > 0.2
    assert correct("absolute-loss-optimum") == [1]


def test_pinball_gives_quantile():
    tau = 0.9
    q = argmin(lambda q: np.mean((y - q) * (tau - (y < q))))
    assert abs(q - np.quantile(y, tau)) < 0.01
    assert Q["pinball-quantile"]["answer"] == round(100 * tau)


def test_bimodal_mean_is_rarely_observed():
    s = np.where(rng.random(20000) < 0.5, -1.0, 1.0) + 0.1 * rng.normal(size=20000)
    assert abs(s.mean()) < 0.05 and np.mean(np.abs(s) < 0.2) < 0.001
    assert correct("bimodal-mean") == [1]


def test_fixed_variance_gaussian_mle_is_least_squares():
    x = rng.uniform(size=500)
    t = 2 * x + 0.3 * rng.normal(size=500)
    ws = np.linspace(1.5, 2.5, 2001)
    nll = [np.mean((t - w * x) ** 2 / (2 * 0.7**2)) + 0.5 * np.log(2 * np.pi * 0.7**2) for w in ws]
    sse = [np.sum((t - w * x) ** 2) for w in ws]
    assert np.argmin(nll) == np.argmin(sse)
    assert correct("distribution-statements") == [0, 1, 2]


def test_overconfident_is_penalized():
    z = rng.normal(size=20000)
    nll = lambda s: np.mean(0.5 * np.log(2 * np.pi * s**2) + z**2 / (2 * s**2))
    assert nll(0.3) > nll(1.0) and nll(3.0) > nll(1.0)
    assert correct("overconfident-score") == [1]
