"""8.2 퀴즈(content/08-calculus-and-autodiff/02-matrix-calculus/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/08-calculus-and-autodiff/02-matrix-calculus/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def num_grad(f, X, eps=1e-6):
    G = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        E = np.zeros_like(X)
        E[idx] = eps
        G[idx] = (f(X + E) - f(X - E)) / (2 * eps)
    return G


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def test_grad_shape_matrix():
    W, x = rng.normal(size=(3, 5)), rng.normal(size=5)
    G = num_grad(lambda M: np.sum((M @ x) ** 2), W)
    assert G.shape == (3, 5)
    W_new = W - 0.01 * G                                # 성분별 업데이트가 바로 맞는다
    assert np.sum((W_new @ x) ** 2) < np.sum((W @ x) ** 2)
    assert correct("grad-shape-matrix") == [0]


def test_quadratic_form_gradient():
    A, x = rng.normal(size=(4, 4)), rng.normal(size=4)
    g = num_grad(lambda v: v @ A @ v, x)
    assert np.allclose(g, (A + A.T) @ x, atol=1e-6)
    assert not np.allclose(g, 2 * A @ x, atol=1e-3)
    assert not np.allclose(g, A.T @ x, atol=1e-3)
    assert correct("quadratic-form-gradient") == [0]


def test_trace_differential():
    M, X = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    g = num_grad(lambda Xv: np.trace(M @ Xv), X)          # dL = tr(M dX)
    assert np.allclose(g, M.T, atol=1e-6)
    assert not np.allclose(g, M, atol=1e-3)
    assert correct("trace-differential") == [0]


def test_logdet_gradient_entry():
    X = np.array([[2.0, 1.0], [0.0, 4.0]])
    g = num_grad(lambda M: np.log(np.linalg.det(M)), X)
    assert np.allclose(g, np.linalg.inv(X).T, atol=1e-6)
    q = Q["logdet-gradient-entry"]
    assert abs(g[1, 0] - q["answer"]) <= q["tolerance"]
    assert abs(np.linalg.inv(X)[1, 0] - q["answer"]) > q["tolerance"]   # 전치를 빠뜨린 오답


def test_softmax_ce_gradient():
    z, y = rng.normal(size=5), np.eye(5)[2]
    g = num_grad(lambda v: -np.sum(y * np.log(softmax(v))), z)
    p = softmax(z)
    assert np.allclose(g, p - y, atol=1e-6)
    assert not np.allclose(g, p * (1 - p), atol=1e-3)
    assert correct("softmax-ce-gradient") == [0]


def test_softmax_ce_uniform():
    z, y = np.zeros(4), np.eye(4)[0]
    g = num_grad(lambda v: -np.sum(y * np.log(softmax(v))), z)
    q = Q["softmax-ce-uniform"]
    assert abs(g[0] - q["answer"]) <= q["tolerance"]
    assert np.isclose(g.sum(), 0, atol=1e-8)


def test_mlp_backprop_transpose():
    K, H = 3, 5
    W2, b2, h = rng.normal(size=(K, H)), rng.normal(size=K), rng.normal(size=H)
    c = rng.normal(size=K)
    L = lambda hv: c @ (W2 @ hv + b2) + 0.5 * np.sum((W2 @ hv + b2) ** 2)
    g_z = c + (W2 @ h + b2)                               # dL/dz
    assert np.allclose(num_grad(L, h), W2.T @ g_z, atol=1e-6)
    assert correct("mlp-backprop-transpose") == [0]
