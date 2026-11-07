"""9.2 퀴즈(content/09-numerical-methods/02-conditioning-and-stability/quiz.toml) 정답을 계산으로 검증한다."""
import math
import tomllib
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np
from scipy.linalg import hilbert

QUIZ = Path(__file__).parents[3] / "content/09-numerical-methods/02-conditioning-and-stability/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
EPS = np.finfo(float).eps
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_cond_or_algorithm():
    # 조건이 나쁜 시스템: solve의 backward error는 eps 수준, forward error는 cond*eps 수준
    A = hilbert(10)
    x = np.ones(10)
    b = A @ x
    errs = {}
    for name, solver in [("lu", np.linalg.solve), ("lstsq", lambda A, b: np.linalg.lstsq(A, b, rcond=None)[0])]:
        xh = solver(A, b)
        bwd = np.linalg.norm(b - A @ xh) / (np.linalg.norm(A, 2) * np.linalg.norm(xh))
        errs[name] = np.linalg.norm(xh - x) / np.linalg.norm(x)
        assert bwd < 10 * EPS
    kappa = np.linalg.cond(A)
    assert errs["lu"] > 1e3 * EPS and errs["lu"] < 10 * kappa * EPS
    assert errs["lstsq"] > 1e3 * EPS                       # 다른 안정한 solver도 eps 수준이 되지 않는다
    assert correct("cond-or-algorithm") == [0]


def test_digits_lost():
    q = Q["digits-lost"]
    assert abs(-math.log10(1e10 * EPS) - q["answer"]) <= q["tolerance"]
    lo, hi, step = q["slider"]["min"], q["slider"]["max"], q["slider"]["step"]
    assert lo <= q["answer"] <= hi and (q["answer"] - lo) % step == 0


def test_quadratic_small_root():
    getcontext().prec = 50
    B = Decimal(10) ** 8
    exact = float((B - (B * B - 4).sqrt()) / 2)
    d = math.sqrt(1e16 - 4)
    naive = (1e8 - d) / 2
    stable = 1 / ((1e8 + d) / 2)
    assert 0.2 < abs(naive - exact) / exact < 0.3                # 약 25%
    assert abs(stable - exact) / exact < 1e-15
    # 조건수 ≈ 1: c를 1e-6만큼 상대적으로 바꾸면 근도 약 1e-6만큼 상대적으로 바뀐다
    c2 = Decimal(1) + Decimal("1e-6")
    root2 = float((B - (B * B - 4 * c2).sqrt()) / 2)
    assert abs((root2 - exact) / exact / 1e-6 - 1) < 1e-3
    assert correct("quadratic-small-root") == [0]


def test_order_from_errors():
    e = np.array([1e-2, 1e-4, 1e-8])
    p = np.log(e[2] / e[1]) / np.log(e[1] / e[0])
    assert abs(p - Q["order-from-errors"]["answer"]) <= Q["order-from-errors"]["tolerance"]


def gd_iters(kappa, tol=1e-8):
    A = np.diag([1.0, kappa])
    step = 2 / (1 + kappa)
    x = np.ones(2)
    k = 0
    while np.linalg.norm(x) > tol * np.sqrt(2):
        x = x - step * (A @ x)
        k += 1
    return k


def test_gd_iterations_kappa():
    ratio = gd_iters(10_000) / gd_iters(100)
    assert 90 < ratio < 110
    assert correct("gd-iterations-kappa") == [0]


def test_scaling_changes_kappa():
    dT = 10 * rng.normal(size=500)              # [K]
    dP = 1e4 * rng.normal(size=500)             # [Pa]
    X = np.c_[dT, dP]                           # (500, 2)
    k0 = np.linalg.cond(X)
    assert 3e2 < k0 < 3e3
    assert np.linalg.cond((X - X.mean(0)) / X.std(0)) < 1.5         # 표준화
    assert np.linalg.cond(np.c_[dT, dP / 1e3]) < 1.5                 # Pa → kPa
    assert np.isclose(np.linalg.cond(1e-3 * X), k0)                  # 전체 상수배: 그대로
    assert np.isclose(np.linalg.cond(X.astype(np.float32)), k0, rtol=1e-4)   # 정밀도: 조건수는 그대로
    assert correct("scaling-changes-kappa") == [0, 1]
