"""10.5 퀴즈(content/10-optimization/05-second-order-methods/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/10-optimization/05-second-order-methods/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_newton_quadratic_one_step():
    R, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    H = R @ np.diag([1.0, 1e6]) @ R.T
    b, x = rng.normal(size=2), rng.normal(size=2)
    steps = 0
    while np.linalg.norm(H @ x - b) > 1e-6 * np.linalg.norm(b) * 1e6 ** 0.5:
        x = x - np.linalg.solve(H, H @ x - b)
        steps += 1
    assert steps == Q["newton-quadratic-one-step"]["answer"] == 1


def test_newton_quadratic_rate():
    e = 1e-2
    assert np.isclose(e**2, 1e-4)
    # 실제 Newton 반복: sqrt(2) 근 찾기에서 오차가 제곱으로 줄어드는지
    x, errs = 2.0, []
    for _ in range(4):
        x = x - (x * x - 2) / (2 * x)
        errs.append(abs(x - np.sqrt(2)))
    ratios = [errs[i + 1] / errs[i] ** 2 for i in range(2)]
    assert all(0.1 < r < 1.0 for r in ratios)                      # e_{k+1} ~ C e_k^2
    assert correct("newton-quadratic-rate") == [1]


def test_newton_saddle():
    g = lambda z: z**3 - z + 0.2
    h = lambda z: 3 * z**2 - 1
    zn, zg = 0.3, 0.3
    for _ in range(200):
        zn, zg = zn - g(zn) / h(zn), zg - 0.1 * g(zg)
    assert abs(g(zn)) < 1e-9 and h(zn) < 0                       # Newton: 곡률 음수인 stationary point
    assert abs(g(zg)) < 1e-9 and h(zg) > 0                       # GD: 국소 최솟점
    assert correct("newton-saddle") == [1]


def test_gauss_newton_approximation():
    t = np.linspace(0, 3, 15)
    y = 2.0 * np.exp(-0.7 * t)
    r = lambda th: th[0] * np.exp(-th[1] * t) - y
    J = lambda th: np.stack([np.exp(-th[1] * t), -th[0] * t * np.exp(-th[1] * t)], axis=1)
    F = lambda th: 0.5 * r(th) @ r(th)

    def num_hess(th, h=1e-4):
        E = np.eye(2) * h
        return np.array([[(F(th + E[i] + E[j]) - F(th + E[i] - E[j]) - F(th - E[i] + E[j]) + F(th - E[i] - E[j]))
                          / (4 * h * h) for j in range(2)] for i in range(2)])

    th_sol = np.array([2.0, 0.7])                                 # 잔차 0인 해
    assert np.allclose(num_hess(th_sol), J(th_sol).T @ J(th_sol), atol=1e-5)
    th_far = np.array([0.5, 2.0])                                 # 잔차가 큰 점
    assert not np.allclose(num_hess(th_far), J(th_far).T @ J(th_far), atol=1e-2)
    JJ = J(th_far).T @ J(th_far)
    assert np.linalg.eigvalsh(JJ).min() >= -1e-12                 # PSD
    assert correct("gauss-newton-approximation") == [0, 1, 2]


def test_lm_damping():
    J, r = rng.normal(size=(10, 3)), rng.normal(size=10)
    g = J.T @ r
    steps = [np.linalg.solve(J.T @ J + lam * np.eye(3), -g) for lam in [1e-2, 1e-1, 1, 10, 100, 1e4]]
    norms = [np.linalg.norm(s) for s in steps]
    cos = [-(s @ g) / (np.linalg.norm(s) * np.linalg.norm(g)) for s in steps]
    assert all(a > b for a, b in zip(norms, norms[1:]))           # 짧아진다
    assert cos[-1] > 0.9999 and cos[-1] > cos[0]                  # gradient 방향으로
    assert correct("lm-damping") == [0]


def test_lbfgs_memory():
    m = 10
    assert 2 * m == Q["lbfgs-memory"]["answer"]


def test_lbfgs_minibatch():
    assert correct("lbfgs-minibatch") == [0]
