"""20.4 퀴즈(content/20-reinforcement-learning/04-bandits-and-offline-rl/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/20-reinforcement-learning/04-bandits-and-offline-rl/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_ucb_bonus():
    t = 55
    ucb = lambda m, n: m + np.sqrt(2 * np.log(t) / n)
    a, b = ucb(0.6, 50), ucb(0.5, 5)
    assert np.isclose(a, 1.00, atol=0.01) and np.isclose(b, 1.77, atol=0.01) and b > a
    assert correct("ucb-bonus") == [1]


def test_thompson_posterior():
    a, b = 1 + 3, 1 + 1
    assert abs(a / (a + b) - Q["thompson-posterior"]["answer"]) <= Q["thompson-posterior"]["tolerance"]


def test_regret_linear():
    p = np.array([0.30, 0.45, 0.50, 0.55, 0.60])
    per_step = 0.1 * np.mean(p.max() - p)           # 학습이 끝난 뒤 ε-greedy의 step당 후회
    assert np.isclose(per_step * 1000, 12.0)
    assert correct("regret-linear") == [0]


def test_offline_overestimate_winners_curse():
    rng = np.random.default_rng(0)
    mu = np.zeros(10)                               # 모든 행동의 참 가치 0
    est = mu + rng.normal(0, 1 / np.sqrt(3), size=(5000, 10))   # 표본 3개의 평균
    chosen = est.max(axis=1)
    assert chosen.mean() > 0.5                      # 고른 행동의 추정은 체계적으로 위로 치우친다
    assert correct("offline-overestimate") == [1]


def test_is_weight_numeric():
    w = np.prod([1.0 / 0.5] * 3)
    assert w == Q["is-weight-numeric"]["answer"] == 8


def test_ois_vs_wis():
    rng = np.random.default_rng(1)
    pb, pe, H = np.array([0.5, 0.5]), np.array([0.2, 0.8]), 20
    o, w_ = [], []
    for _ in range(200):
        a = rng.choice(2, size=(1000, H), p=pb)
        G = a.sum(axis=1)
        w = np.prod(pe[a] / pb[a], axis=1)
        o.append(np.mean(w * G))
        w_.append(np.sum(w * G) / np.sum(w))
    assert np.std(o) > 5 * np.std(w_)               # WIS 분산이 훨씬 작다
    assert abs(np.mean(w_) - 0.8 * H) > 0.1         # WIS는 편향
    assert correct("ois-vs-wis") == [0, 1]


def test_constrained_lagrangian():
    rew, cost, budget = np.array([0.2, 0.5, 0.8, 0.9]), np.array([0.0, 0.1, 0.4, 0.9]), 0.25
    def avg_cost(lam):
        return cost[np.argmax(rew - lam * cost)]
    assert avg_cost(0.5) > budget and avg_cost(2.0) <= budget   # λ를 키우면 비용이 줄어든다
    assert correct("constrained-lagrangian") == [0]
