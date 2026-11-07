"""12.2 퀴즈(content/12-statistical-inference/02-likelihood/quiz.toml) 정답을 계산으로 검증한다."""
import math
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/12-statistical-inference/02-likelihood/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_likelihood_not_density():
    theta = np.linspace(0, 1, 200001)
    lik = math.comb(20, 7) * theta**7 * (1 - theta) ** 13
    assert abs(lik.mean() - 1 / 21) < 1e-4 and abs(lik.mean() - 0.048) < 0.001
    assert correct("likelihood-not-density") == [0]


def test_exponential_mle():
    t = np.array([2.0, 3.0, 5.0, 10.0])
    lams = np.linspace(0.01, 1.0, 99001)
    ll = len(t) * np.log(lams) - lams * t.sum()
    assert abs(lams[np.argmax(ll)] - Q["exponential-mle"]["answer"]) < 1e-4


def test_why_log_likelihood():
    t = rng.exponential(2.0, size=1000)
    assert np.prod(0.5 * np.exp(-0.5 * t)) == 0.0                      # underflow
    X = rng.normal(size=(20000, 5))
    assert abs(X.var(axis=1).mean() - 0.8) < 0.02                       # 로그우도로 구한 MLE도 편향
    assert correct("why-log-likelihood") == [0, 1, 2]


def test_mse_gaussian_statements():
    x = rng.uniform(size=400)
    y = 1.5 * x + rng.exponential(size=400) - 1.0                       # 가우시안이 아닌 잡음, 평균 0
    ws = np.linspace(1.0, 2.0, 1001)
    mse = [np.mean((y - w * x) ** 2) for w in ws]
    nll = [np.mean(0.5 * np.log(2 * np.pi * 0.3**2) + (y - w * x) ** 2 / (2 * 0.3**2)) for w in ws]
    assert np.argmin(mse) == np.argmin(nll) and not np.isclose(min(mse), min(nll))
    assert correct("mse-gaussian-statements") == [0, 1]


def test_l1_laplace():
    y = rng.laplace(1.0, 0.5, size=3001)
    grid = np.linspace(0, 2, 2001)
    nll = [np.mean(np.log(2 * 0.5) + np.abs(y - c) / 0.5) for c in grid]
    assert abs(grid[np.argmin(nll)] - np.median(y)) < 2e-3
    assert correct("l1-laplace") == [1]


def test_zero_failure_mle():
    theta = np.linspace(0, 1, 10001)
    assert theta[np.argmax(theta**3)] == 1.0
    post = theta**3 * theta * (1 - theta)                               # Beta(2,2) 사전분포 × 우도
    assert abs(theta[np.argmax(post)] - 0.8) < 1e-3
    assert correct("zero-failure-mle") == [0]


def test_ridge_as_prior():
    X = rng.normal(size=(30, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.3 * rng.normal(size=30)
    sigma, tau = 0.3, 0.5
    w = np.linalg.solve(X.T @ X + sigma**2 / tau**2 * np.eye(4), X.T @ y)
    grad = -X.T @ (y - X @ w) / sigma**2 + w / tau**2                  # 음의 로그 사후분포의 gradient
    assert np.allclose(grad, 0, atol=1e-9)
    assert correct("ridge-as-prior") == [0]
