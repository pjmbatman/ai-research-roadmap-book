"""14.3 퀴즈(content/14-information-and-learning-theory/03-objective-design/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/14-information-and-learning-theory/03-objective-design/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_known_noise_weights():
    # 1/σ² 가중치가 다른 가중치보다 분산이 작다(가우스–마르코프의 가중 버전)
    rng = np.random.default_rng(0)
    x = np.linspace(0.1, 1, 30)
    sigma = np.where(np.arange(30) % 3 == 0, 0.05, 0.8)
    est = {k: [] for k in ("inv_var", "none", "inv_sd", "var")}
    w = {"inv_var": 1 / sigma**2, "none": np.ones(30), "inv_sd": 1 / sigma, "var": sigma**2}
    for _ in range(3000):
        y = 2 * x + sigma * rng.normal(size=30)
        for k in est:
            est[k].append(np.sum(w[k] * x * y) / np.sum(w[k] * x * x))
    sd = {k: np.std(v) for k, v in est.items()}
    assert sd["inv_var"] == min(sd.values())
    assert correct("known-noise-weights") == [0]


def test_nll_difference_kl():
    p = np.array([0.5, 0.3, 0.2])
    qa, qb = np.array([0.3, 0.4, 0.3]), np.array([0.45, 0.35, 0.2])
    kl = lambda q: np.sum(p * np.log(p / q))
    ce = lambda q: -np.sum(p * np.log(q))
    assert np.isclose(ce(qa) - ce(qb), kl(qa) - kl(qb))
    assert correct("nll-difference-kl") == [0]


def test_elbo_gap():
    q = Q["elbo-gap"]
    assert abs((-2.20 - 0.25) - q["answer"]) <= q["tolerance"]
    # 항등식 확인: log p(x) = ELBO + KL
    prior, mu, x = np.array([0.3, 0.7]), np.array([-1.0, 2.0]), 0.0
    lj = np.log(prior) - 0.5 * (x - mu) ** 2 - 0.5 * np.log(2 * np.pi)
    lpx = np.log(np.exp(lj).sum())
    post = np.exp(lj - lpx)
    qz = np.array([0.2, 0.8])
    assert np.isclose(np.sum(qz * (lj - np.log(qz))) + np.sum(qz * np.log(qz / post)), lpx)


def test_contrastive_baseline():
    K = 128
    S = np.zeros((K, K))
    logp = S - np.log(np.exp(S).sum(1, keepdims=True))
    loss = -np.mean(np.diag(logp))
    q = Q["contrastive-baseline"]
    assert np.isclose(loss, np.log(K)) and abs(loss - q["answer"]) <= q["tolerance"]


def test_unit_dependent_sum():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 50)
    yA = 10 * x + rng.normal(0, 0.5, 50)
    yB = 5000 * 1.1 * x + rng.normal(0, 200, 50)
    w = lambda a, ya, b, yb: (a * x @ ya + b * x @ yb) / ((a**2 + b**2) * (x @ x))
    w_pa, w_kpa = w(10, yA, 5000, yB), w(10, yA, 5.0, yB * 1e-3)
    wA_only = (x @ yA) / (10 * x @ x)
    assert abs(w_kpa - wA_only) < abs(w_pa - wA_only)            # kPa에서 온도 과제 쪽으로
    assert correct("unit-dependent-sum") == [0]


def test_padding_mean():
    loss = np.array([[1.0, 1, 1, 1], [1, 1, 0, 0]])
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=bool)
    assert np.isclose(loss.mean(), Q["padding-mean"]["answer"])
    assert np.isclose(loss.sum() / mask.sum(), 1.0)


def test_sum_reduction_lr():
    rng = np.random.default_rng(0)
    x = rng.normal(size=128)
    # 손실 Σ_i (w x_i)²의 w = 1에서의 gradient = Σ_i 2 x_i²
    g32, g128 = np.sum(2 * x[:32] ** 2), np.sum(2 * x[:128] ** 2)
    assert 2.5 < g128 / g32 < 6                                  # 대략 4배
    assert correct("sum-reduction-lr") == [0]
