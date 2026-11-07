"""7.1 퀴즈(content/07-linear-algebra/01-vector-spaces/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial as P

QUIZ = Path(__file__).parents[3] / "content/07-linear-algebra/01-vector-spaces/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_subspace_test():
    # 0: 합이 0인 평면 — 무작위 원소의 선형결합도 조건을 만족
    u, v = rng.normal(size=3), rng.normal(size=3)
    u, v = u - u.mean(), v - v.mean()
    assert np.isclose((2 * u - 3 * v).sum(), 0)
    # 1: 합이 1인 평면은 영벡터를 포함하지 않는다
    assert np.zeros(3).sum() != 1
    # 2: x'' + x = 0의 해 cos, sin의 선형결합도 해(유한차분으로 확인)
    t, h = np.linspace(0, 3, 301), 1e-3
    f = lambda s: 2 * np.cos(s) - 5 * np.sin(s)
    assert np.allclose((f(t + h) - 2 * f(t) + f(t - h)) / h**2 + f(t), 0, atol=1e-4)
    # 3: 차수가 정확히 2인 다항식 두 개의 합이 차수 2가 아닐 수 있다
    s = np.array([1.0, 0, 1]) + np.array([2.0, 1, -1])
    assert np.trim_zeros(s, "b").size - 1 != 2
    # 4: x1 >= 0 집합은 -1배에 닫혀 있지 않다
    assert (-1 * np.array([1.0, 0]))[0] < 0
    assert correct("subspace-test") == [0, 2]


def test_independence_check():
    V = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 2]], dtype=float).T
    assert np.linalg.matrix_rank(V) == 2
    assert np.allclose(V[:, 0] + V[:, 1], V[:, 2])
    assert correct("independence-check") == [1]


def test_polynomial_dimension():
    x = np.linspace(-1, 1, 10)
    monomials = np.stack([x**j for j in range(4)], axis=1)   # (10, 4)
    assert np.linalg.matrix_rank(monomials) == Q["polynomial-dimension"]["answer"] == 4


def test_which_maps_linear():
    x = np.linspace(0, 1, 2001)
    f, g, a, b = np.sin(3 * x), x**2, 2.0, -0.7
    integral = lambda y: np.trapezoid(y, x)
    at_half = lambda y: y[1000]
    for T in (integral, at_half):
        assert np.isclose(T(a * f + b * g), a * T(f) + b * T(g))
    p, q = rng.normal(size=4), rng.normal(size=4)
    assert np.allclose(P.polyder(a * p + b * q), a * P.polyder(p) + b * P.polyder(q))
    shift = lambda v: v + np.array([1.0, 0.0])
    assert not np.allclose(shift(np.zeros(2)), 0)
    v = rng.normal(size=2)
    assert not np.isclose(np.linalg.norm(-v), -np.linalg.norm(v))
    assert correct("which-maps-linear") == [0, 1, 2]


def test_matrix_columns():
    A = np.array([[2.0, -1.0], [1.0, 3.0]])
    assert np.allclose(A @ [1, 0], [2, 1]) and np.allclose(A @ [0, 1], [-1, 3])
    assert not np.allclose(A.T @ [1, 0], [2, 1])
    assert correct("matrix-columns") == [0]


def test_derivative_kernel_dim():
    D = np.diag(np.arange(1.0, 4), k=1)                      # (4, 4)
    assert 4 - np.linalg.matrix_rank(D) == Q["derivative-kernel-dim"]["answer"] == 1
    assert np.allclose(D @ [1.0, 0, 0, 0], 0)                # 상수는 핵에 있다
