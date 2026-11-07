"""10.1 퀴즈(content/10-optimization/01-optimization-basics/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/10-optimization/01-optimization-basics/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def midpoint_violations(f, lo=-3, hi=3, n=20000, dim=1):
    """f((a+b)/2) > (f(a)+f(b))/2 인 무작위 쌍의 수. 0보다 크면 볼록이 아니다."""
    a, b = rng.uniform(lo, hi, size=(2, n, dim))
    return int(np.sum(f((a + b) / 2) > (f(a) + f(b)) / 2 + 1e-12))


def test_unconstrained_min_infeasible():
    g = np.linspace(0, 1, 1001)
    X1, X2 = np.meshgrid(g, g)
    F = np.where(X1 + X2 <= 1 + 1e-12, (X1 - 2) ** 2 + (X2 - 1) ** 2, np.inf)
    i = np.unravel_index(np.argmin(F), F.shape)
    assert np.allclose([X1[i], X2[i]], [1, 0]) and np.isclose(F[i], 2.0)
    f = lambda p: (p[0] - 2) ** 2 + (p[1] - 1) ** 2
    assert f((0.5, 0.5)) > 2 and f((0, 0)) > 2
    assert correct("unconstrained-min-infeasible") == [1]


def test_hessian_diagonal_not_enough():
    lam = np.linalg.eigvalsh(np.array([[2.0, 4.0], [4.0, 2.0]]))
    assert np.allclose(lam, [-2, 6])
    f = lambda p: p[..., 0] ** 2 + 4 * p[..., 0] * p[..., 1] + p[..., 1] ** 2
    assert midpoint_violations(f, dim=2) > 0
    assert correct("hessian-diagonal-not-enough") == [2]


def test_convexity_rules():
    f = lambda x: x[..., 0] ** 2
    g = lambda x: (x[..., 0] - 1) ** 2
    A, b = np.array([[2.0]]), np.array([0.5])
    assert midpoint_violations(lambda x: f(x) + g(x)) == 0
    assert midpoint_violations(lambda x: np.maximum(f(x), g(x))) == 0
    assert midpoint_violations(lambda x: f(x @ A.T + b)) == 0
    assert midpoint_violations(lambda x: f(x) * g(x)) > 0                 # 곱: 이중 우물
    assert midpoint_violations(lambda x: 0 * f(x) - f(x)) > 0             # 차: 오목
    assert correct("convexity-rules") == [0, 1, 2]


def test_convex_without_minimizer():
    x = np.linspace(-30, 30, 10001)
    fp = 1 / (1 + np.exp(-x))                                             # f'(x) = sigmoid(x)
    assert np.all(fp > 0)                                                 # stationary point 없음
    f = np.logaddexp(0, x)
    assert np.all(np.diff(f) > 0) and f[0] > 0
    assert correct("convex-without-minimizer") == [1]


def test_tilted_double_well_minima():
    r = np.sort(np.roots([1, 0, -1, 0.2]).real)
    assert len(r) == 3
    curv = 3 * r**2 - 1
    assert int(np.sum(curv > 0)) == Q["tilted-double-well-minima"]["answer"] == 2
    f = r**4 / 4 - r**2 / 2 + 0.2 * r
    assert np.argmin(f) == 0                                              # 왼쪽 최솟점이 전역


def test_second_order_inconclusive():
    h = 0.1
    cases = {"x^3": lambda x: x**3, "x^4": lambda x: x**4, "-x^4": lambda x: -(x**4)}
    kinds = set()
    for f in cases.values():
        lo, hi = f(-h), f(h)
        kinds.add("min" if lo > 0 and hi > 0 else "max" if lo < 0 and hi < 0 else "neither")
    assert kinds == {"min", "max", "neither"}
    assert correct("second-order-inconclusive") == [2]


def test_convex_stationary_is_global():
    # f(x, y) = x^2: y축 전체가 전역 최솟점 -> 유일하지 않다
    pts = np.array([[0.0, -1.0], [0.0, 0.0], [0.0, 3.0]])
    assert np.allclose(pts[:, 0] ** 2, 0)
    assert correct("convex-stationary-is-global") == [0]
