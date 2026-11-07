"""9.4 퀴즈(content/09-numerical-methods/04-nonlinear-equations/quiz.toml) 정답을 계산으로 검증한다."""
import math
import tomllib
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

QUIZ = Path(__file__).parents[3] / "content/09-numerical-methods/04-nonlinear-equations/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
f = lambda x: x**3 - 2 * x - 5
ROOT = brentq(f, 2, 3, xtol=1e-15)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_bisection_iterations():
    k = next(k for k in range(1, 60) if 2.0**-k <= 1e-6)
    q = Q["bisection-iterations"]
    assert k == q["answer"] == 20
    assert q["slider"]["min"] <= k <= q["slider"]["max"]


def test_secant_efficiency():
    x0, x1, err = 3.0, 2.5, []
    for _ in range(6):
        x0, x1 = x1, x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        err.append(abs(x1 - ROOT))
    e = np.log(err)
    p = (e[2:] - e[1:-1]) / (e[1:-1] - e[:-2])
    assert abs(p[-1] - (1 + 5**0.5) / 2) < 0.05
    assert math.sqrt(2) < (1 + 5**0.5) / 2                     # 호출 1번당 효율: Newton √2 < secant φ
    assert correct("secant-efficiency") == [0]


def test_fixed_point_rearrangement():
    gs = [lambda x: (2 * x + 5) ** (1 / 3), lambda x: (x**3 - 5) / 2, lambda x: 5 / (x**2 - 2)]
    dg = lambda g: abs((g(ROOT + 1e-6) - g(ROOT - 1e-6)) / 2e-6)
    slopes = [dg(g) for g in gs]
    assert slopes[0] < 1 < slopes[1] and slopes[2] > 1
    assert abs(slopes[0] - 0.152) < 1e-3 and abs(slopes[1] - 6.58) < 1e-2 and abs(slopes[2] - 3.7) < 0.05
    for g, converges in zip(gs, [True, False, False]):
        x = 2.0
        with np.errstate(all="ignore"):
            for _ in range(40):
                x = g(x)
                if not np.isfinite(x) or abs(x) > 1e6:
                    break
        assert (abs(x - ROOT) < 1e-10) == converges
    assert correct("fixed-point-rearrangement") == [0]


def test_double_root_linear():
    g = lambda x: (x - 1) ** 2 * (x + 2)
    dg = lambda x: 2 * (x - 1) * (x + 2) + (x - 1) ** 2
    x, err = 1.5, []
    for _ in range(20):
        x = x - g(x) / dg(x)
        err.append(abs(x - 1))
    ratios = np.array(err[1:]) / np.array(err[:-1])
    assert abs(ratios[-1] - 0.5) < 0.01
    assert correct("double-root-linear") == [0]


def test_newton_arctan_damping():
    x, path = 1.5, [1.5]
    for _ in range(4):
        x = x - np.arctan(x) * (1 + x**2)
        path.append(x)
    assert np.allclose(np.round(path, 2), [1.5, -1.69, 2.32, -5.11, 32.3], atol=0.01)
    x = 1.5
    for _ in range(4):
        step, t = np.arctan(x) * (1 + x**2), 1.0
        while abs(np.arctan(x - t * step)) >= abs(np.arctan(x)) and t > 1e-4:
            t /= 2
        x -= t * step
    assert abs(x) < 1e-9
    assert correct("newton-arctan-damping") == [0]


def test_implicit_euler_newton():
    lam = np.array([[-1000.0, 0.0], [0.0, -1.0]])             # stiff 선형계 + 비선형 항
    fun = lambda y: lam @ y + 0.1 * y**2
    jac = lambda y: lam + np.diag(0.2 * y)
    y_n, h = np.array([1.0, 1.0]), 0.1
    y = y_n.copy()                                             # 직전 값을 초기값으로
    for k in range(10):
        G = y - y_n - h * fun(y)
        if np.linalg.norm(G) < 1e-12:
            break
        y = y - np.linalg.solve(np.eye(2) - h * jac(y), G)    # Jacobian = I - h J_f
    assert np.linalg.norm(y - y_n - h * fun(y)) < 1e-12 and k <= 6
    assert correct("implicit-euler-newton") == [0]
