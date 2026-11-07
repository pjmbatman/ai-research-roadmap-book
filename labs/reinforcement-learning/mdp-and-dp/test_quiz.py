"""20.1 퀴즈(content/20-reinforcement-learning/01-mdp-and-dp/quiz.toml) 정답을 계산으로 검증한다."""
import math
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/20-reinforcement-learning/01-mdp-and-dp/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_discount_return():
    g = sum(0.9**k for k in range(2000))
    assert abs(g - Q["discount-return"]["answer"]) <= Q["discount-return"]["tolerance"]


def test_vi_iterations():
    k = next(k for k in range(1, 1000) if 0.9**k <= 1e-3)
    assert k == Q["vi-iterations"]["answer"] == 66
    assert math.ceil(math.log(1e-3) / math.log(0.99)) == 688       # 해설의 gamma = 0.99 값


def test_bellman_linear_vs_max():
    # 정책 평가는 선형계 한 번, 최적 가치는 max 때문에 반복이 필요하다.
    rng = np.random.default_rng(0)
    S, A, g = 6, 3, 0.9
    P = rng.dirichlet(np.ones(S), size=(S, A))
    R = rng.normal(size=(S, A))
    pi = rng.dirichlet(np.ones(A), size=S)
    P_pi, r_pi = np.einsum("sa,sat->st", pi, P), (pi * R).sum(axis=1)
    V = np.linalg.solve(np.eye(S) - g * P_pi, r_pi)
    assert np.allclose(V, r_pi + g * P_pi @ V)
    T = lambda v: (R + g * P @ v).max(axis=1)
    u, w = rng.normal(size=S), rng.normal(size=S)
    assert not np.allclose(T(u + w) - T(np.zeros(S)), (T(u) - T(np.zeros(S))) + (T(w) - T(np.zeros(S))))  # 비선형
    assert correct("bellman-linear-vs-max") == [0]


def test_policy_iteration_finite():
    rng = np.random.default_rng(1)
    S, A, g = 5, 3, 0.9
    P = rng.dirichlet(np.ones(S), size=(S, A))
    R = rng.normal(size=(S, A))
    pi, values = np.zeros(S, dtype=int), []
    for _ in range(50):
        V = np.linalg.solve(np.eye(S) - g * P[np.arange(S), pi], R[np.arange(S), pi])
        values.append(V)
        new = (R + g * P @ V).argmax(axis=1)
        if np.array_equal(new, pi):
            break
        pi = new
    for a, b in zip(values, values[1:]):
        assert np.all(b >= a - 1e-12)                               # 상태마다 개선(선택지 2가 틀림)
    for V0 in (np.zeros(S), 100 * rng.normal(size=S)):              # 초기값 무관(선택지 1)
        v = V0
        for _ in range(500):
            v = (R + g * P @ v).max(axis=1)
        assert np.allclose(v, values[-1])
    assert correct("policy-iteration-finite") == [0, 1]


def test_pomdp_belief_example():
    door = np.array([0, 1, 0, 1, 0])
    b = np.full(5, 0.2) * np.where(door == 1, 0.9, 0.1)
    b /= b.sum()
    assert np.isclose(b[1], b[3]) and np.isclose(b[1], 0.429, atol=5e-4)   # 관측 하나로는 위치가 정해지지 않음
    assert correct("pomdp-belief") == [1]


def test_markov_state():
    # 각도만으로는 다음 각도가 정해지지 않는다: 같은 각도, 다른 각속도.
    dt, theta = 0.1, 0.2
    nxt = [theta + dt * w for w in (+1.0, -1.0)]
    assert nxt[0] != nxt[1]
    assert correct("markov-state") == [1]
