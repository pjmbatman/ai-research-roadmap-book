"""11.3 퀴즈(content/11-probability/03-conditional-probability/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/11-probability/03-conditional-probability/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)
P = np.array([[0.3, 0.1],
              [0.2, 0.4]])                     # p(x, y), 행 x, 열 y


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def close(qid, value):
    return abs(Q[qid]["answer"] - value) <= Q[qid]["tolerance"]


def test_marginal_from_joint():
    assert close("marginal-from-joint", P[:, 1].sum())


def test_conditional_from_joint():
    assert close("conditional-from-joint", P[1, 1] / P[:, 1].sum())


def test_base_rate_posterior():
    prev, sens, fpr = 0.01, 0.9, 0.09
    post = sens * prev / (sens * prev + fpr * (1 - prev))
    assert close("base-rate-posterior", 100 * post)
    n = 2_000_000                               # 시뮬레이션으로도 확인
    sick = rng.random(n) < prev
    pos = np.where(sick, rng.random(n) < sens, rng.random(n) < fpr)
    assert abs(sick[pos].mean() - post) < 0.005


def test_inverse_conditional():
    prev, recall, fpr = 0.001, 0.99, 0.01
    precision = recall * prev / (recall * prev + fpr * (1 - prev))
    assert precision < 0.1
    assert correct("inverse-conditional") == [0]


def test_uncorrelated_not_independent():
    x = rng.uniform(-1, 1, 400_000)
    y = x**2
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.01
    assert np.mean(y > 0.5) < 0.35 and np.all(y[np.abs(x) > 0.8] > 0.5)
    assert correct("uncorrelated-not-independent") == [1]


def test_collider_explaining_away():
    n = 1_000_000
    leak, drift = rng.random(n) < 0.05, rng.random(n) < 0.10
    alarm = leak | drift
    assert leak[alarm & drift].mean() < leak[alarm].mean() - 0.2
    assert correct("collider-explaining-away") == [1]


def test_chain_factorization():
    # A -> B -> C 에서 A ⊥ C | B 이지만 A, C는 주변적으로 의존한다
    n = 1_000_000
    a = rng.random(n) < 0.5
    b = rng.random(n) < np.where(a, 0.8, 0.2)
    c = rng.random(n) < np.where(b, 0.9, 0.1)
    assert abs(c[a].mean() - c[~a].mean()) > 0.3
    for bv in (False, True):
        m = b == bv
        assert abs(c[m & a].mean() - c[m & ~a].mean()) < 0.01
    assert correct("chain-factorization") == [0]
