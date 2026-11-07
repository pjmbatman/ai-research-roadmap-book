"""10.2 퀴즈(content/10-optimization/02-constrained-optimization/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/10-optimization/02-constrained-optimization/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)
P, A_VEC = np.array([2.0, 1.0]), np.array([1.0, 1.0])


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def project(c):
    mu = max(0.0, 2 * (A_VEC @ P - c) / (A_VEC @ A_VEC))
    return P - mu / 2 * A_VEC, mu


def test_lagrange_gradient_parallel():
    # min x^2 + 2y^2 s.t. x + y = 1 -> (2/3, 1/3)
    K = np.array([[2.0, 0, 1], [0, 4.0, 1], [1, 1, 0]])
    x, y, nu = np.linalg.solve(K, [0, 0, 1.0])
    gf, gh = np.array([2 * x, 4 * y]), np.array([1.0, 1.0])
    assert np.linalg.norm(gf) > 1                                 # grad f != 0
    assert np.isclose(gf[0] * gh[1] - gf[1] * gh[0], 0)           # 평행
    assert nu < 0                                                 # 등식 승수는 음수일 수도 있다
    assert correct("lagrange-gradient-parallel") == [1]


def test_multiplier_value():
    x, mu = project(0.5)
    assert np.isclose(mu, Q["multiplier-value"]["answer"])
    f = lambda c: np.sum((project(c)[0] - P) ** 2)
    assert np.isclose((f(0.5 + 1e-6) - f(0.5 - 1e-6)) / 2e-6, -mu, atol=1e-5)
    # 격자 탐색으로 투영 확인
    z = rng.uniform(-3, 3, size=(200000, 2))
    z = z[z @ A_VEC <= 0.5]
    assert np.min(np.sum((z - P) ** 2, axis=1)) >= np.sum((x - P) ** 2) - 1e-9


def test_inactive_constraint():
    x, mu = project(4.0)
    assert mu == 0 and np.allclose(x, P) and A_VEC @ x < 4
    assert correct("inactive-constraint") == [0, 1]


def test_weak_duality():
    # cell 1의 문제: g(nu) = -3 nu^2 / 8 - nu, p* = 2/3
    nus = rng.uniform(-10, 10, size=10000)
    assert np.all(-3 * nus**2 / 8 - nus <= 2 / 3 + 1e-12)
    assert correct("weak-duality") == [0]


def test_penalty_not_exact():
    viol, conds = [], []
    for rho in [1.0, 1e2, 1e4, 1e6]:
        H = np.array([[2 + rho, rho], [rho, 4 + rho]])
        x, y = np.linalg.solve(H, [rho, rho])
        viol.append(x + y - 1)
        conds.append(np.linalg.cond(H))
        assert np.isclose(x + y - 1, -4 / (4 + 3 * rho))
    assert all(v != 0 for v in viol)
    assert conds[-1] / conds[-2] > 50                              # 조건수가 rho에 비례
    assert correct("penalty-not-exact") == [1]


def test_ridge_radius():
    X, y = rng.normal(size=(60, 5)), rng.normal(size=60)
    norms = [np.linalg.norm(np.linalg.solve(X.T @ X / 60 + lam * np.eye(5), X.T @ y / 60)) for lam in [0.01, 0.1, 1, 10]]
    assert all(a > b for a, b in zip(norms, norms[1:]))
    assert correct("ridge-radius") == [0]


def test_l1_exact_zeros():
    v, t = rng.normal(size=1000), 0.5
    soft = np.sign(v) * np.maximum(np.abs(v) - t, 0)
    ridge = v / (1 + t)
    assert np.sum(soft == 0) == np.sum(np.abs(v) <= t) > 0
    assert np.sum(ridge == 0) == 0
    assert correct("l1-exact-zeros") == [0]
