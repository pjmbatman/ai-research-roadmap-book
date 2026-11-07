"""12.6 퀴즈(content/12-statistical-inference/06-regression-models/quiz.toml) 정답을 계산으로 검증한다."""
import math
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/12-statistical-inference/06-regression-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def ols(X, y):
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    r = y - X @ b
    return b, np.sqrt(r @ r / (len(y) - X.shape[1]) * np.diag(XtX_inv))


def test_ols_assumptions():
    x = np.linspace(0, 1, 40)
    X = np.column_stack([np.ones_like(x), x])
    slopes, ses = [], []
    for _ in range(4000):
        eps = rng.exponential(1.0, 40) - 1.0                       # 정규가 아니지만 평균 0
        eps *= 0.2 + 2.0 * x                                       # 이분산
        b, s = ols(X, 1.0 + 2.0 * x + eps)
        slopes.append(b[1]); ses.append(s[1])
    assert abs(np.mean(slopes) - 2.0) < 0.05                       # 여전히 불편
    assert abs(np.mean(ses) / np.std(slopes) - 1) > 0.1            # 표준 공식의 SE는 틀린다
    assert correct("ols-assumptions") == [0, 1]


def test_poisson_rate_ratio():
    assert abs(math.exp(0.693) - Q["poisson-rate-ratio"]["answer"]) <= Q["poisson-rate-ratio"]["tolerance"]


def test_logistic_coefficient():
    sig = lambda z: 1 / (1 + math.exp(-z))
    odds = lambda p: p / (1 - p)
    for b0 in (-4.0, 0.0, 2.0):
        assert math.isclose(odds(sig(b0 + 0.4)) / odds(sig(b0)), math.exp(0.4))
    assert abs((sig(0.4) - sig(0.0)) - 0.1) < 0.002
    assert correct("logistic-coefficient") == [1]


def test_residual_funnel():
    assert correct("residual-funnel") == [1]


def test_durbin_watson():
    e = np.zeros(20000)
    for t in range(1, e.size):
        e[t] = 0.6 * e[t - 1] + rng.normal()
    dw = np.sum(np.diff(e) ** 2) / np.sum(e**2)
    assert abs(dw - 0.8) < 0.03
    assert math.isclose(Q["durbin-watson"]["answer"], 2 * (1 - 0.6))


def test_design_effect():
    G, m, rho = 10, 20, 0.5
    means = []
    for _ in range(4000):
        y = np.repeat(rng.normal(0, math.sqrt(rho), G), m) + rng.normal(0, math.sqrt(1 - rho), G * m)
        means.append(y.mean())
    deff = np.var(means) / (1.0 / (G * m))                        # 실제 분산 / 독립 가정 분산
    assert abs(deff - 10.5) < 0.8
    assert math.isclose(Q["design-effect"]["answer"], 1 + (m - 1) * rho)


def test_high_leverage_point():
    assert correct("high-leverage-point") == [1]
