"""14.5 퀴즈(content/14-information-and-learning-theory/05-modern-generalization/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

QUIZ = Path(__file__).parents[3] / "content/14-information-and-learning-theory/05-modern-generalization/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def min_norm_risks(n, ps, D=200, noise=0.1, reps=15, lam=0.0):
    beta = np.arange(1, D + 1) ** -0.5
    beta /= np.linalg.norm(beta)
    out = []
    for p in ps:
        rs = []
        for s in range(reps):
            r = np.random.default_rng(s)
            X = r.normal(size=(n, D))
            y = X @ beta + noise * r.normal(size=n)
            Xp = X[:, :p]
            w = np.linalg.pinv(Xp) @ y if lam == 0 else Xp.T @ np.linalg.solve(Xp @ Xp.T + lam * np.eye(n), y)
            rs.append(np.sum((w - beta[:p]) ** 2) + np.sum(beta[p:] ** 2) + noise**2)
        out.append(np.median(rs))
    return np.array(out)


def test_double_descent_peak():
    ps = np.array([30, 40, 46, 50, 54, 60, 80])
    risks = min_norm_risks(50, ps, D=120, reps=7)
    q = Q["double-descent-peak"]
    assert abs(ps[np.argmax(risks)] - q["answer"]) <= q["tolerance"]


def test_ridge_removes_peak():
    ps = [10, 30, 40, 50, 120]
    plain, ridge = min_norm_risks(40, ps, reps=7), min_norm_risks(40, ps, reps=7, lam=1.0)
    assert plain.max() > 10 * ridge.max()
    assert correct("ridge-removes-peak") == [0]


def test_gd_min_norm():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(10, 40)), rng.normal(size=10)
    w = np.zeros(40)
    lr = 1 / np.linalg.norm(X, 2) ** 2
    for _ in range(5000):
        w -= lr * X.T @ (X @ w - y)
    assert np.allclose(w, np.linalg.pinv(X) @ y, atol=1e-8)
    assert correct("gd-min-norm") == [0]


def test_lazy_regime_and_compute_optimal():
    assert correct("lazy-regime") == [0]
    assert correct("compute-optimal") == [0]


def test_scaling_floor():
    # 바닥이 있는 곡선을 작은 범위에서 순수 멱법칙으로 맞추면 먼 곳을 과소 예측한다
    N = np.array([25, 50, 100, 200, 3200])
    L = 0.05 + 0.3 * N**-0.6
    b, log_a = np.polyfit(np.log(N[:4]), np.log(L[:4]), 1)
    assert np.exp(log_a) * 3200**b < L[4]
    assert correct("scaling-floor") == [0]


def _frac_in_hull(n_train, d, n_test=60, seed=0):
    rng = np.random.default_rng(seed)
    train, test = rng.normal(size=(n_train, d)), rng.normal(size=(n_test, d))
    A = np.vstack([train.T, np.ones(n_train)])
    ok = [linprog(np.zeros(n_train), A_eq=A, b_eq=np.append(q, 1.0), bounds=(0, None), method="highs").status == 0
          for q in test]
    return np.mean(ok)


def test_high_dim_extrapolation():
    assert _frac_in_hull(500, 1) > 0.9
    assert _frac_in_hull(500, 16) < 0.1
    assert _frac_in_hull(1000, 16) < 0.2
    assert correct("high-dim-extrapolation") == [0, 1]
