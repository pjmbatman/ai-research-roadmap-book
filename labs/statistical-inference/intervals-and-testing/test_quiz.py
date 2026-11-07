"""12.4 퀴즈(content/12-statistical-inference/04-intervals-and-testing/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from math import erf, sqrt
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/12-statistical-inference/04-intervals-and-testing/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)
Phi = np.vectorize(lambda z: 0.5 * (1 + erf(z / sqrt(2))))


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_ci_meaning():
    X = rng.normal(0.112, 0.02, size=(20000, 30))
    m, se = X.mean(axis=1), X.std(axis=1, ddof=1) / np.sqrt(30)
    cover = np.mean(np.abs(m - 0.112) <= 2.045 * se)                    # 절차의 포함 확률
    new = rng.normal(0.112, 0.02, size=20000)
    inside = np.mean(np.abs(new - m[0]) <= 2.045 * se[0])               # 개별 측정값이 들어가는 비율
    assert abs(cover - 0.95) < 0.01 and inside < 0.5
    assert correct("ci-meaning") == [0]


def test_p_value_meaning():
    a, b = rng.normal(0, 1, (20000, 20)), rng.normal(0, 1, (20000, 20))  # 귀무가설이 참
    z = (b.mean(axis=1) - a.mean(axis=1)) / np.sqrt(2 / 20)
    p = 2 * (1 - Phi(np.abs(z)))
    assert abs(np.mean(p <= 0.03) - 0.03) < 0.005                        # H0 아래에서 P(p ≤ 0.03) = 0.03
    assert correct("p-value-meaning") == [1]


def test_significance_vs_importance():
    n, d = 100000, 0.02
    a, b = rng.normal(0, 1, n), rng.normal(d, 1, n)
    se = np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
    assert abs((b.mean() - a.mean()) / se) > 3.29                        # p < 0.001
    assert correct("significance-vs-importance") == [0, 1]


def test_bootstrap_unit():
    assert correct("bootstrap-unit") == [1]


def test_fwer_ten_tests():
    exact = 1 - 0.95**10
    sim = np.mean((rng.random((200000, 10)) < 0.05).any(axis=1))      # H0 아래 p-value는 균등분포
    assert abs(sim - exact) < 0.005
    assert abs(Q["fwer-ten-tests"]["answer"] - 100 * exact) <= Q["fwer-ten-tests"]["tolerance"]


def test_bh_controls_fdr():
    m, m1 = 100, 10
    fdp, any_false = [], []
    for _ in range(2000):
        z = rng.normal(np.r_[np.full(m1, 3.0), np.zeros(m - m1)], 1.0)
        p = 2 * (1 - Phi(np.abs(z)))
        order = np.argsort(p)
        ok = np.nonzero(p[order] <= 0.05 * np.arange(1, m + 1) / m)[0]
        rej = np.zeros(m, dtype=bool)
        if ok.size:
            rej[order[: ok.max() + 1]] = True
        fdp.append(rej[m1:].sum() / max(rej.sum(), 1))
        any_false.append(rej[m1:].any())
    assert np.mean(fdp) <= 0.05 and np.mean(any_false) > 0.1
    assert correct("bh-controls-fdr") == [0]
