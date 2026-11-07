"""24.1 퀴즈(content/24-scientific-ml/01-physics-and-learning/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

QUIZ = Path(__file__).parents[3] / "content/24-scientific-ml/01-physics-and-learning/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def pendulum(t, y, g, L, c):
    return [y[1], -(g / L) * np.sin(y[0]) - c * y[1]]


def test_length_scale_time():
    t = 5 * np.sqrt(4.0 / 9.81)
    q = Q["length-scale-time"]
    assert abs(t - q["answer"]) <= q["tolerance"]
    # 무차원 궤적과 실제 궤적이 같은지 확인한다
    T0 = np.sqrt(4.0 / 9.81)
    dim = solve_ivp(pendulum, (0, 5 * T0), [1.0, 0.0], args=(9.81, 4.0, 0.0), rtol=1e-10, atol=1e-12).y[0, -1]
    nondim = solve_ivp(pendulum, (0, 5), [1.0, 0.0], args=(1.0, 1.0, 0.0), rtol=1e-10, atol=1e-12).y[0, -1]
    assert abs(dim - nondim) < 1e-8


def test_scaling_gradient_descent():
    rng = np.random.default_rng(0)
    n = 200
    temp = rng.uniform(280, 360, n)
    conc = rng.uniform(1e-4, 2e-3, n)
    y = 0.02 * (temp - 300) + 800 * conc + rng.normal(0, 0.05, n)

    def gd(X):
        H = X.T @ X / n
        lr = 1.0 / np.linalg.eigvalsh(H).max()
        w = np.zeros(3)
        for _ in range(5000):
            w -= lr * X.T @ (X @ w - y) / n
        return w, np.linalg.cond(H)

    raw, cond_raw = gd(np.column_stack([np.ones(n), temp, conc]))
    std, cond_std = gd(np.column_stack([np.ones(n), (temp - temp.mean()) / temp.std(), (conc - conc.mean()) / conc.std()]))
    assert cond_raw > 1e10 and cond_std < 2
    assert abs(raw[2]) < 1.0                                  # 원 단위: 농도 계수를 배우지 못함
    assert np.all(np.isfinite(raw))                           # 발산한 것이 아니다
    assert abs(std[2] / conc.std() - 805.7) < 1.0             # 표준화: 최소제곱 해와 같음
    assert correct("scaling-gradient-descent") == [1]


def test_error_sources_noise_shrinks():
    rng = np.random.default_rng(1)
    g, L = 9.81, 1.0
    T = 2 * np.pi * np.sqrt(L / g)

    def l_err(m):
        errs = []
        for _ in range(200):
            Lh = g * ((T + rng.normal(0, 0.01, m)).mean() / (2 * np.pi)) ** 2
            errs.append(abs(Lh - L))
        return np.mean(errs)

    assert l_err(1000) < l_err(10) / 5                        # 무작위 잡음 오차는 데이터로 준다
    assert correct("error-sources") == [1, 2, 3]


def test_conservation_structure():
    # 위치만의 함수인 힘 + Verlet: 에너지 오차가 step 수에 따라 자라지 않는다
    a = np.array([-1.002, 0.176, -0.012])                     # 본문 셀 3에서 학습된 계수
    dt = 0.1
    th = [1.0, 1.0 + 0.5 * dt**2 * a.sum()]
    for _ in range(20000):
        th.append(2 * th[-1] - th[-2] + dt**2 * (a @ [th[-1], th[-1] ** 3, th[-1] ** 5]))
    th = np.array(th)
    E = 0.5 * ((th[2:] - th[:-2]) / (2 * dt)) ** 2 - np.cos(th[1:-1])
    E0 = -np.cos(1.0)
    early, late = np.abs(E[:1000] - E0).max(), np.abs(E[:20000] - E0).max()
    assert late < 1.1 * early                                 # 유계
    assert correct("conservation-structure") == [1]


def test_other_single_answers():
    assert correct("problem-kind") == [2]
    assert correct("physics-injection-routes") == [1]
