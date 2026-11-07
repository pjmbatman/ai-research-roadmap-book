"""14.4 퀴즈(content/14-information-and-learning-theory/04-generalization-basics/quiz.toml) 정답을 계산으로 검증한다."""
import itertools
import math
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/14-information-and-learning-theory/04-generalization-basics/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def sample(rng, n, noise=0.3):
    x = rng.uniform(0, 1, n)
    return x, np.sin(2 * np.pi * x) + noise * rng.normal(size=n)


def test_training_risk_optimism():
    rng = np.random.default_rng(0)
    x_te, y_te = sample(rng, 20000)
    emp, pop = [], []
    for _ in range(200):
        x, y = sample(rng, 15)
        c = np.polyfit(x, y, 7)
        emp.append(np.mean((np.polyval(c, x) - y) ** 2))
        pop.append(np.mean((np.polyval(c, x_te) - y_te) ** 2))
    assert np.mean(emp) < 0.09 < np.median(pop)
    assert correct("training-risk-optimism") == [0]


def test_error_decomposition():
    assert correct("error-decomposition") == [0, 1]


def test_bias_variance_degree():
    assert 0.156 + 0.032 + 0.09 < 0.001 + 0.708 + 0.09
    assert correct("bias-variance-degree") == [0]


def test_hoeffding_sample_size():
    n = math.ceil(math.log(2 / 0.05) / (2 * 0.05**2))
    assert n == Q["hoeffding-sample-size"]["answer"]
    assert 2 * math.exp(-2 * n * 0.05**2) <= 0.05 < 2 * math.exp(-2 * (n - 1) * 0.05**2)


def _separable(points, labels):
    """작은 선형계획 대신, 무작위 방향 탐색으로 선형 분리 가능 여부를 판정한다."""
    rng = np.random.default_rng(0)
    X = np.column_stack([points, np.ones(len(points))])
    W = rng.normal(size=(200000, 3))
    ok = np.all(np.sign(X @ W.T) == labels[:, None], axis=0)
    return bool(ok.any())


def test_vc_dimension_plane():
    three = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert all(_separable(three, np.array(l)) for l in itertools.product([-1, 1], repeat=3))
    four = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert not _separable(four, np.array([1, 1, -1, -1]))    # XOR 배치
    assert Q["vc-dimension-plane"]["answer"] == 3


def test_no_free_lunch():
    assert correct("no-free-lunch") == [0]


def test_random_labels_capacity():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 150))
    y = rng.choice([-1.0, 1.0], size=100)
    w = np.linalg.pinv(X) @ y
    assert np.mean(np.sign(X @ w) == y) == 1.0                # 무작위 레이블을 모두 맞춘다
    assert correct("random-labels-capacity") == [0]
