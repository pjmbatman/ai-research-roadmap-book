"""19.2 퀴즈(content/19-generative-and-foundation-models/02-autoregressive-models/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from itertools import product
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/19-generative-and-foundation-models/02-autoregressive-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_chain_rule_exact():
    P = rng.dirichlet(np.ones(8)).reshape(2, 2, 2)
    for order in [(0, 1, 2), (1, 2, 0), (2, 1, 0)]:
        for x in product([0, 1], repeat=3):
            prob, fixed = 1.0, {}
            for i in order:
                m = lambda a: sum(P[s] for s in product([0, 1], repeat=3) if all(s[j] == v for j, v in a.items()))
                prob *= m({**fixed, i: x[i]}) / m(fixed)
                fixed[i] = x[i]
            assert np.isclose(prob, P[x])
    assert correct("chain-rule-exact") == [0]


def test_ar_loglik_sum():
    q = Q["ar-loglik-sum"]
    assert abs(np.log(0.5 * 0.2 * 0.8) - q["answer"]) <= q["tolerance"]


def test_causal_mask_parallel():
    assert correct("causal-mask-parallel") == [0]


def test_nucleus_size():
    p = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
    n_keep = int(np.searchsorted(np.cumsum(p), 0.8)) + 1
    assert n_keep == Q["nucleus-size"]["answer"] == 3


def test_temperature_limit():
    z = np.array([2.0, 1.0, 0.5])
    soft = lambda T: np.exp(z / T - (z / T).max()) / np.exp(z / T - (z / T).max()).sum()
    assert soft(0.01).argmax() == 0 and soft(0.01)[0] > 0.999      # greedy
    assert np.allclose(soft(1e6), 1 / 3, atol=1e-5)                 # 균등
    assert np.all(np.argsort(soft(0.3)) == np.argsort(soft(3.0)))   # 순위 보존
    assert correct("temperature-limit") == [0]


def test_exposure_bias_symptom():
    # 진폭이 스스로 제한되는 과정에 선형 AR(2)를 맞추면 1-step 오차는 작지만 자유 생성 진폭이 커진다
    def sim(n, R, g):
        x = np.zeros((R, n)); x[:, 0], x[:, 1] = 1.0, 0.95
        for t in range(2, n):
            a, b = x[:, t - 1], x[:, t - 2]
            x[:, t] = 1.9 * a - 0.95 * b + 0.15 * (a - a**3) + 0.02 * g.normal(size=R)
        return x
    x = sim(3000, 1, np.random.default_rng(1))[0]
    X = np.stack([x[1:-1], x[:-2]], 1)
    w = np.linalg.lstsq(X, x[2:], rcond=None)[0]
    sig = (x[2:] - X @ w).std()
    assert sig < 0.05
    g = np.random.default_rng(2)
    xs = np.zeros((200, 400)); xs[:, 0], xs[:, 1] = 1.0, 0.95
    for t in range(2, 400):
        xs[:, t] = np.stack([xs[:, t - 1], xs[:, t - 2]], 1) @ w + sig * g.normal(size=200)
    truth = sim(400, 200, np.random.default_rng(3))
    q = lambda a: np.quantile(np.abs(a[:, 200:]), 0.95)
    assert q(xs) > 1.3 * q(truth)
    assert correct("exposure-bias-symptom") == [0]
