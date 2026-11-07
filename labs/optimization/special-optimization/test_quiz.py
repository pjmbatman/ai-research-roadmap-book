"""10.7 퀴즈(content/10-optimization/07-special-optimization/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.stats import norm

QUIZ = Path(__file__).parents[3] / "content/10-optimization/07-special-optimization/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def ei(mu, sd, best):
    z = (best - mu) / sd
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


def test_pareto_dominance():
    F = np.array([[1, 5], [2, 3], [3, 3], [4, 1], [2, 4]], float)
    nd = [i for i, p in enumerate(F) if not np.any(np.all(F <= p, 1) & np.any(F < p, 1))]
    assert correct("pareto-dominance") == nd == [0, 1, 3]


def test_weighted_sum_nonconvex():
    f1 = np.linspace(0, 1, 1001)
    f2 = 1 - f1**2
    picks = {float(f1[np.argmin(w * f1 + (1 - w) * f2)]) for w in np.linspace(0, 1, 201)}
    assert picks <= {0.0, 1.0}
    assert np.isclose(0.5 * 0.5 + 0.5 * (1 - 0.25), 0.625)
    assert correct("weighted-sum-nonconvex") == [0]


def test_implicit_hypergradient():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(20, 5)), rng.normal(size=20)
    Xv, yv = rng.normal(size=(30, 5)), rng.normal(size=30)
    w = lambda lam: np.linalg.solve(X.T @ X + lam * np.eye(5), X.T @ y)
    L = lambda lam: 0.5 * np.mean((Xv @ w(lam) - yv) ** 2)
    lam = 0.7
    dw = -np.linalg.solve(X.T @ X + lam * np.eye(5), w(lam))
    hg = (Xv.T @ (Xv @ w(lam) - yv) / 30) @ dw
    assert np.isclose(hg, (L(lam + 1e-6) - L(lam - 1e-6)) / 2e-6, rtol=1e-5)
    assert correct("implicit-hypergradient") == [0]


def test_grid_distinct_values():
    g = np.linspace(0, 1, 4)
    pts = np.array([[a, b] for a in g for b in g])
    assert len(np.unique(pts[:, 0])) == Q["grid-distinct-values"]["answer"]
    r = np.random.default_rng(0).random((16, 2))
    assert len(np.unique(r[:, 0])) == 16


def test_ei_exploration():
    p, q = ei(0.5, 0.1, 0.0), ei(0.5, 1.0, 0.0)
    assert p < 1e-6 and np.isclose(q, 0.198, atol=1e-3)
    assert np.isclose(norm.cdf(-0.5), 0.31, atol=0.01)
    assert correct("ei-exploration") == [0]


def test_ei_at_incumbent():
    val = ei(0.0, 1.0, 0.0)
    assert abs(val - Q["ei-at-incumbent"]["answer"]) <= Q["ei-at-incumbent"]["tolerance"]
    assert np.isclose(ei(0.0, 2.0, 0.0), 2 * val)
    rng = np.random.default_rng(1)
    assert abs(np.mean(np.maximum(-rng.normal(size=400000), 0)) - val) < 0.005


def test_lp_relaxation_bound():
    value = np.array([10, 13, 7, 8, 9, 4, 6, 11.0])
    cost = np.array([5, 7, 4, 5, 6, 3, 4, 6.0])
    budget = 21.0
    lp = linprog(-value, A_ub=cost[None, :], b_ub=[budget], bounds=[(0, 1)] * 8)
    res = milp(-value, constraints=LinearConstraint(cost[None, :], 0, budget), integrality=np.ones(8), bounds=Bounds(0, 1))
    brute = max(value @ np.array(c) for c in product([0, 1], repeat=8) if cost @ np.array(c) <= budget)
    assert -lp.fun >= -res.fun - 1e-9 and np.isclose(-res.fun, brute)
    assert cost @ np.round(lp.x) > budget                          # 반올림은 infeasible
    assert value @ np.floor(lp.x + 1e-9) < brute                    # 내림은 최적이 아님
    assert correct("lp-relaxation-bound") == [0]
