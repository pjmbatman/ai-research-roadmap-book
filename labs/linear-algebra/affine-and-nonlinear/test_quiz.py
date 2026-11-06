"""7.2 퀴즈(content/07-linear-algebra/02-affine-and-nonlinear/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/07-linear-algebra/02-affine-and-nonlinear/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_affine_is_not_linear():
    A, b = rng.normal(size=(2, 2)), np.array([0.3, -0.1])
    f = lambda x: A @ x + b
    assert not np.allclose(f(np.zeros(2)), 0)
    x, y = rng.normal(size=2), rng.normal(size=2)
    assert not np.allclose(f(x + y), f(x) + f(y))
    assert correct("affine-is-not-linear") == [1]


def test_stacked_affine_layers():
    W1, b1, W2, b2 = rng.normal(size=(4, 2)), rng.normal(size=4), rng.normal(size=(3, 4)), rng.normal(size=3)
    x = rng.normal(size=2)
    g = W2 @ (W1 @ x + b1) + b2
    assert np.allclose(g, W2 @ W1 @ x + W2 @ b1 + b2)          # 선택지 0
    assert not np.allclose(g, W2 @ W1 @ x + b1[:3] + b2)       # 선택지 1(b1에 W2를 빠뜨림; shape 맞추려 앞 3개)
    assert correct("stacked-affine-layers") == [0]


def test_affine_combinations():
    coefs = [(0.3, 0.7), (1.5, -0.5), (0.5, 0.5), (2, 1)]
    expected = [i for i, c in enumerate(coefs) if np.isclose(sum(c), 1)]
    assert correct("affine-combinations") == expected


def test_relu_linear_regions():
    x = np.linspace(-3, 5, 80001)
    f = np.maximum(0, x) + np.maximum(0, x - 1) + np.maximum(0, x - 2)
    slope = np.round(np.diff(f) / np.diff(x), 6)
    pieces = len({s for s in slope})
    assert pieces == Q["relu-linear-regions"]["answer"] == 4


def test_relu_extrapolation_is_affine_far_out():
    W1, b1, w2 = rng.normal(size=32), rng.normal(size=32), rng.normal(size=32)
    net = lambda x: np.maximum(0, np.outer(x, W1) + b1) @ w2
    x = np.array([20.0, 30.0, 40.0])                          # 모든 꺾이는 점 -b/w보다 오른쪽
    assert np.all(-b1 / W1 < 20)
    y = net(x)
    assert np.isclose(y[2] - y[1], y[1] - y[0])               # 직선
    assert correct("relu-extrapolation") == [0]


def test_homogeneous_coordinates():
    A, b, x = rng.normal(size=(2, 2)), rng.normal(size=2), rng.normal(size=2)
    xh = np.append(x, 1.0)
    good = np.block([[A, b[:, None]], [np.zeros((1, 2)), np.ones((1, 1))]])
    bad = np.block([[A, np.zeros((2, 1))], [b[None, :], np.ones((1, 1))]])
    assert np.allclose((good @ xh)[:2], A @ x + b)
    assert not np.allclose((bad @ xh)[:2], A @ x + b)
    assert correct("homogeneous-coordinates") == [0]
