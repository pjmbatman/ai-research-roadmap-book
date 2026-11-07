"""9.7 퀴즈(content/09-numerical-methods/07-pde-and-dae/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

QUIZ = Path(__file__).parents[3] / "content/09-numerical-methods/07-pde-and-dae/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def ftcs_growth(r, N=20, steps=400):
    x = np.linspace(0, 1, N + 1)
    u = np.sin(np.pi * x) + 1e-6 * np.sin((N - 1) * np.pi * x)
    for _ in range(steps):
        u[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
    return np.max(np.abs(u))


def test_ftcs_dt_limit():
    dx_m, alpha_m2ps = 0.01, 1e-4
    dt_max = dx_m**2 / (2 * alpha_m2ps)
    q = Q["ftcs-dt-limit"]
    assert abs(dt_max - q["answer"]) <= q["tolerance"]
    assert ftcs_growth(0.5) <= 1.0 and ftcs_growth(0.55) > 1.0   # r = 1/2가 경계


def test_explicit_heat_cost():
    def cost(N, alpha=1.0, T=0.1):
        dt = (1 / N) ** 2 / (2 * alpha)
        return (N + 1 - 2) * int(np.ceil(T / dt))               # 내부점 수 × step 수
    ratio = cost(400) / cost(200)
    assert abs(ratio - Q["explicit-heat-cost"]["answer"]) < 0.1


def test_cfl_meaning():
    M = 100
    x = np.arange(M) / M
    def run(nu):
        u = np.exp(-200 * (x - 0.3) ** 2)
        for _ in range(round(1.0 / (nu / M))):
            u = u - nu * (u - np.roll(u, 1))
        return u
    assert run(0.5).max() < 0.6                                   # 안정하지만 부정확(수치 확산)
    assert run(1.1).max() > 1.0                                   # 불안정
    assert correct("cfl-meaning") == [1]


def test_dae_index_pendulum():
    # 제약 c = x²+y²-L²: c' = 2(x vx + y vy)에는 λ가 없고, c'' = 2(vx²+vy²) + 2(x ax + y ay)에서 처음 λ가 나타난다.
    # c'' = 0을 λ에 대해 푼 식이 λ = (v² - g y)/(x²+y²)이고, λ의 ODE를 얻으려면 한 번 더 미분 → index 3.
    g, rng = 9.81, np.random.default_rng(0)
    th, om = rng.uniform(-1, 1, 2)
    x, y, vx, vy = np.sin(th), -np.cos(th), om * np.cos(th), om * np.sin(th)   # 일관된 상태
    c1 = 2 * (x * vx + y * vy)
    assert abs(c1) < 1e-12                                        # 속도 제약(숨은 제약)
    for lam in (0.0, 1.0, 5.0):                                   # c''는 λ에 의존한다
        ax, ay = -lam * x, -lam * y - g
        c2 = 2 * (vx**2 + vy**2) + 2 * (x * ax + y * ay)
        lam_star = (vx**2 + vy**2 - g * y) / (x**2 + y**2)
        assert np.isclose(c2, -2 * (x**2 + y**2) * (lam - lam_star))
    assert Q["dae-index-pendulum"]["answer"] == 3


def test_dae_consistent_init():
    g = 9.81
    def rhs(t, s):
        x, y, vx, vy = s
        lam = (vx**2 + vy**2 - g * y) / (x**2 + y**2)
        return [vx, vy, -lam * x, -lam * y - g]
    sol = solve_ivp(rhs, (0, 100), [1.0, 0.0, 0.1, 0.0], rtol=1e-8, atol=1e-11)
    x, y, vx, vy = sol.y[:, -1]
    assert np.isclose(x**2 + y**2 - 1.0, 2 * 0.1 * 100, rtol=1e-3)   # 잔차 = 2(x·v)_0 t
    assert correct("dae-consistent-init") == [1]


def test_observed_order():
    errs = [4.0e-3, 1.0e-3, 2.5e-4]
    ps = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    q = Q["observed-order"]
    assert all(abs(p - q["answer"]) <= q["tolerance"] for p in ps)


def test_loose_tolerance_plateau():
    def err(N, rtol, atol, T=0.05):
        x = np.linspace(0, 1, N + 1)[1:-1]
        L = diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(N - 1, N - 1)) / (1.0 / N) ** 2
        ex = lambda t: np.exp(-np.pi**2 * t) * np.sin(np.pi * x) + 0.5 * np.exp(-9 * np.pi**2 * t) * np.sin(3 * np.pi * x)
        sol = solve_ivp(lambda t, u: L @ u, (0, T), ex(0), method="BDF", jac=L, rtol=rtol, atol=atol)
        return np.max(np.abs(sol.y[:, -1] - ex(T)))
    loose = [err(N, 1e-3, 1e-6) for N in (80, 160)]
    tight = [err(N, 1e-10, 1e-12) for N in (80, 160)]
    assert loose[0] / loose[1] < 1.5                              # 멈춤
    assert 3.8 < tight[0] / tight[1] < 4.2                        # 2차 수렴
    assert correct("loose-tolerance-plateau") == [1]
