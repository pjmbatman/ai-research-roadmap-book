"""14.2 퀴즈(content/14-information-and-learning-theory/02-mutual-information/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/14-information-and-learning-theory/02-mutual-information/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def H(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def mi(pxy):
    return H(pxy.sum(1)) + H(pxy.sum(0)) - H(pxy.ravel())


def test_zero_correlation_mi():
    xs, ys = np.array([-1, 0, 1]), np.array([0, 1])
    pxy = np.array([[1 / 3 * (y == x**2) for y in ys] for x in xs])
    cov = np.sum(pxy * np.outer(xs, ys)) - np.sum(pxy.sum(1) * xs) * np.sum(pxy.sum(0) * ys)
    assert abs(cov) < 1e-12
    assert np.isclose(mi(pxy), H(pxy.sum(0))) and mi(pxy) < H(pxy.sum(1))
    assert correct("zero-correlation-mi") == [0]


def test_data_processing():
    rng = np.random.default_rng(0)
    pxy = np.full(4, 0.25)[:, None] * (0.7 * np.eye(4) + 0.1)
    for _ in range(50):                                        # 임의의 처리 p(z | y)
        ch = rng.dirichlet(np.ones(3), size=4)
        assert mi(pxy @ ch) <= mi(pxy) + 1e-12
    # 독립적인 두 번째 센서: (Y, Y2)는 Y보다 정보가 많을 수 있다
    pxyy = np.einsum("xa,xb->xab", pxy, (0.7 * np.eye(4) + 0.1)).reshape(4, 16)
    assert mi(pxyy) > mi(pxy)
    assert correct("data-processing") == [0, 1, 2]


def test_histogram_bias():
    vals = []
    for seed in range(10):
        x, y = np.random.default_rng(seed).normal(size=(2, 200))   # 독립
        c, _, _ = np.histogram2d(x, y, bins=50)
        vals.append(mi(c / c.sum()))
    assert np.mean(vals) > 1.0
    assert correct("histogram-bias") == [0]


def test_infonce_ceiling():
    assert abs(np.log(256) - Q["infonce-ceiling"]["answer"]) <= Q["infonce-ceiling"]["tolerance"]
    # 어떤 점수 행렬이든 InfoNCE ≤ log K
    rng = np.random.default_rng(0)
    K = 256
    S = rng.normal(size=(K, K)) * 50
    S[np.arange(K), np.arange(K)] += 1000                      # 진짜 짝을 압도적으로 높게
    lse = np.log(np.mean(np.exp(S - S.max(1, keepdims=True)), axis=1)) + S.max(1)
    nce = np.mean(np.diag(S) - lse)
    assert nce <= np.log(K) + 1e-9 and nce > np.log(K) - 1e-6


def test_contrastive_batch_size():
    assert np.log(256) > 5.2 and np.log(4096) > 7.9
    assert np.log(256) - 5.2 < 0.5 and np.log(4096) - 7.9 < 0.5   # 둘 다 천장 근처
    assert correct("contrastive-batch-size") == [0]


def test_deterministic_ib():
    # 결정적 함수의 bin MI는 bin 수에 따라 끝없이 커진다(연속 X에서 I(X; f(X))는 유한하지 않다)
    x = np.random.default_rng(0).uniform(size=20000)
    t = np.tanh(3 * x)
    vals = []
    for b in (4, 16, 64):
        c, _, _ = np.histogram2d(x, t, bins=b)
        vals.append(mi(c / c.sum()))
    assert vals[0] < vals[1] < vals[2]
    assert correct("deterministic-ib") == [0]
