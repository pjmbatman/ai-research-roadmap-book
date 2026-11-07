"""9.6 퀴즈(content/09-numerical-methods/06-ode-solvers/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

QUIZ = Path(__file__).parents[3] / "content/09-numerical-methods/06-ode-solvers/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def robertson(t, y):
    y1, y2, y3 = y
    return [-0.04 * y1 + 1e4 * y2 * y3, 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2, 3e7 * y2**2]


def test_euler_halving():
    def euler(h, T=10.0):
        y = np.array([1.0, 0.0])
        for _ in range(round(T / h)):
            y = y + h * np.array([y[1], -y[0]])
        return np.linalg.norm(y - [np.cos(T), -np.sin(T)])
    ratios = [euler(0.01) / euler(0.005), euler(0.005) / euler(0.0025)]
    q = Q["euler-halving"]
    assert all(abs(r - q["answer"]) <= q["tolerance"] for r in ratios)


def test_implicit_step_cost():
    bdf = solve_ivp(robertson, (0, 1), [1, 0, 0], method="BDF")
    rk = solve_ivp(robertson, (0, 1), [1, 0, 0], method="RK45")
    assert bdf.njev > 0 and rk.njev == 0                      # implicit은 Jacobian을 쓴다
    assert bdf.t.size * 30 < rk.t.size                        # 대신 step이 훨씬 적다
    assert correct("implicit-step-cost") == [1]


def test_euler_stability_limit():
    lam = -50.0
    h_max = Q["euler-stability-limit"]["answer"]
    assert np.isclose(h_max, -2 / lam)
    for h, grows in [(0.039, False), (0.041, True)]:
        y = 1.0
        for _ in range(2000):
            y = (1 + h * lam) * y
        assert (abs(y) > 1) == grows


def test_stiffness_signs():
    y = np.array([0.9, 3e-5, 0.1])
    J = np.array([[-0.04, 1e4 * y[2], 1e4 * y[1]],
                  [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
                  [0.0, 6e7 * y[1], 0.0]])
    ev = np.sort(np.abs(np.linalg.eigvals(J).real))
    nonzero = ev[ev > 1e-12]
    assert nonzero.max() / nonzero.min() > 1e3                # 여러 자릿수의 시간 척도
    # 빠른 진동(비stiff): explicit과 implicit의 step 수가 비슷하다 → 선택지 3은 stiffness의 신호가 아니다
    osc = lambda t, y: [100 * y[1], -100 * y[0]]
    n_rk = solve_ivp(osc, (0, 1), [1, 0], method="RK45", rtol=1e-6).t.size
    n_bdf = solve_ivp(osc, (0, 1), [1, 0], method="BDF", rtol=1e-6).t.size
    assert n_bdf > n_rk / 3
    assert correct("stiffness-signs") == [0, 1, 2]


def test_implicit_large_step_accuracy():
    lam, h = -50.0, 1.0
    y1 = 1.0 / (1 - h * lam)                                  # implicit Euler 한 step
    assert abs(y1) < 1                                        # 발산하지 않지만
    assert y1 / np.exp(lam * h) > 1e18                        # 정확해 e^{-50}과 전혀 다르다
    assert correct("implicit-large-step-accuracy") == [1]


def test_default_tolerance():
    doc = solve_ivp.__doc__
    assert "1e-3" in doc and "1e-6" in doc                    # 기본값 rtol=1e-3, atol=1e-6
    sol = solve_ivp(lambda t, y: [y[1], -y[0]], (0, 100), [1.0, 0.0])
    assert abs(sol.y[0, -1] - np.cos(100)) > 10 * 1e-3       # 전역 오차가 rtol보다 훨씬 크다
    assert correct("default-tolerance") == [1]


def test_atol_small_quantity():
    y30 = solve_ivp(lambda t, y: -y, (0, 30), [1.0]).y[0, -1]
    assert abs(y30 / np.exp(-30) - 1) > 1e3                   # 기본 atol에서 작은 값은 제어되지 않는다
    y30_tight = solve_ivp(lambda t, y: -y, (0, 30), [1.0], atol=1e-20).y[0, -1]
    assert abs(y30_tight / np.exp(-30) - 1) < 0.1
    assert correct("atol-small-quantity") == [1]
