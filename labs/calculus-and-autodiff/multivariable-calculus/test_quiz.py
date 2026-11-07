"""8.1 퀴즈(content/08-calculus-and-autodiff/01-multivariable-calculus/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/08-calculus-and-autodiff/01-multivariable-calculus/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_partials_not_differentiable():
    f = lambda x, y: 0.0 if x == 0 and y == 0 else x * y / (x**2 + y**2)
    h = 1e-8
    assert abs((f(h, 0) - f(0, 0)) / h) < 1e-12 and abs((f(0, h) - f(0, 0)) / h) < 1e-12   # 편미분 0
    assert np.isclose(f(1e-9, 1e-9), 0.5) and f(1e-9, 0.0) == 0.0                       # 방향마다 극한이 다르다
    assert correct("partials-not-differentiable") == [1]


def test_gradient_steepest_direction():
    f = lambda p: p[0] ** 2 + 3 * p[1] ** 2
    p, h = np.array([1.0, 1.0]), 1e-7
    th = np.linspace(0, 2 * np.pi, 36000, endpoint=False)
    U = np.stack([np.cos(th), np.sin(th)], axis=1)
    D = np.array([(f(p + h * u) - f(p)) / h for u in U])
    best = U[np.argmax(D)]
    choices = [np.array([1, 3]) / np.sqrt(10), np.array([1, 1]) / np.sqrt(2), np.array([-3, 1]) / np.sqrt(10), np.array([0, 1])]
    dists = [np.linalg.norm(best - c) for c in choices]
    assert int(np.argmin(dists)) == 0 and dists[0] < 1e-3
    assert correct("gradient-steepest-direction") == [0]


def test_hessian_curvature():
    lam = np.array([0.4, 100.0])
    V = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    H = V @ np.diag(lam) @ V.T
    c = 1.0
    radius = [np.sqrt(2 * c / (v @ H @ v)) for v in V.T]   # 등고선 0.5 h^T H h = c의 반지름
    assert radius[0] > radius[1]                            # 작은 고윳값 방향으로 길다
    assert np.isclose(radius[0] / radius[1], np.sqrt(250))
    assert np.all(np.linalg.eigvalsh(H) > 0)
    assert correct("hessian-curvature") == [0]


def test_chain_rule_order():
    A1, A2 = rng.normal(size=(4, 3)), rng.normal(size=(2, 4))
    f = lambda x: np.tanh(A1 @ x)                 # R^3 -> R^4
    g = lambda u: A2 @ np.sin(u)                  # R^4 -> R^2
    x, eps = rng.normal(size=3), 1e-6
    J_h = np.stack([(g(f(x + eps * e)) - g(f(x - eps * e))) / (2 * eps) for e in np.eye(3)], axis=1)
    J_f = np.diag(1 - np.tanh(A1 @ x) ** 2) @ A1  # (4, 3)
    J_g = A2 @ np.diag(np.cos(f(x)))              # (2, 4), f(x)에서 계산
    assert np.allclose(J_h, J_g @ J_f, atol=1e-6)
    assert J_h.shape == (2, 3)
    assert correct("chain-rule-order") == [0]


def test_chain_rule_two_paths():
    u = lambda s, t: (s + t) * (s - t)
    h = 1e-6
    d = (u(2 + h, 1) - u(2 - h, 1)) / (2 * h)
    assert abs(d - Q["chain-rule-two-paths"]["answer"]) <= Q["chain-rule-two-paths"]["tolerance"] + 1e-6


def test_taylor_quadratic_model():
    B = rng.normal(size=(3, 3))
    H, g = B @ B.T + 3 * np.eye(3), rng.normal(size=3)
    model = lambda h: g @ h + 0.5 * h @ H @ h
    h_star = -np.linalg.solve(H, g)
    for cand in (-g, -H @ g, np.zeros(3)):
        assert model(h_star) < model(cand)
    for _ in range(100):
        assert model(h_star) <= model(h_star + 1e-2 * rng.normal(size=3))
    assert correct("taylor-quadratic-model") == [0]


def test_taylor_error_scaling():
    f = lambda t: np.exp(t)                       # 2계 도함수가 0이 아닌 함수
    err = lambda d: abs(f(d) - (1 + d))
    scale = 0.02 / err(0.1)                       # 거리 0.1에서 오차 0.02가 되도록 크기만 조정
    q = Q["taylor-error-scaling"]
    assert abs(scale * err(0.05) - q["answer"]) <= q["tolerance"]
