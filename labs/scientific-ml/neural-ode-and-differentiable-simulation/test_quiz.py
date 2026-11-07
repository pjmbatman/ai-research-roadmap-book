"""24.3 퀴즈(content/24-scientific-ml/03-neural-ode-and-differentiable-simulation/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

QUIZ = Path(__file__).parents[3] / "content/24-scientific-ml/03-neural-ode-and-differentiable-simulation/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def pendulum(t, y, p):
    return [y[1], -p[0] * np.sin(y[0]) - p[1] * y[1]]


T_OBS = np.arange(0.1, 5.01, 0.1)
_rng = np.random.default_rng(0)
OBS = solve_ivp(pendulum, (0, 5), [1.0, 0.0], t_eval=T_OBS, args=([9.81, 0.3],), rtol=1e-10, atol=1e-12).y[0]
OBS = OBS + _rng.normal(0, 0.01, T_OBS.size)


def loss(p, rtol):
    th = solve_ivp(pendulum, (0, 5), [1.0, 0.0], t_eval=T_OBS, args=(p,), rtol=rtol, atol=rtol * 1e-2).y[0]
    return np.mean((th - OBS) ** 2)


def fd_dc(rtol, h):
    p0 = np.array([8.0, 0.5])
    e = np.array([0.0, 1.0])
    return (loss(p0 + h * e, rtol) - loss(p0 - h * e, rtol)) / (2 * h)


def test_loose_tolerance_gradient():
    loose = [fd_dc(1e-3, h) for h in (1e-2, 1e-6)]
    tight = fd_dc(1e-10, 1e-6)
    assert abs(loose[0] - loose[1]) < 1e-3                     # h에 거의 무관
    assert abs(loose[1] - tight) / abs(tight) > 0.05           # 참 gradient와 5% 이상 차이
    assert correct("loose-tolerance-gradient") == [1]


def test_explicit_gradient_limit():
    h = 0.05
    limit = 2 / h
    q = Q["explicit-gradient-limit"]
    assert abs(limit - q["answer"]) <= q["tolerance"]

    def final(kappa):
        y = 0.0
        for n in range(40):
            y = y - h * kappa * (y - np.cos(n * h))
        return y
    assert abs(final(39.0)) < 10 and abs(final(41.0)) > 1      # 경계를 넘으면 커진다
    assert abs(final(60.0)) > 1e10


def test_single_and_multiple_answers():
    assert correct("adjoint-cost") == [1]
    assert correct("backprop-vs-adjoint-memory") == [0]
    assert correct("ude-structure") == [0]
    assert correct("ude-extrapolation") == [0, 1, 2]


def test_ude_damping_velocity_range():
    # 1 rad에서 놓은 진자의 최대 각속도는 약 2.7 rad/s, 2 rad에서는 약 4.5 rad/s다
    damping = lambda om: 0.2 * om + 0.1 * om * np.abs(om)
    f = lambda t, y: [y[1], -9.81 * np.sin(y[0]) - damping(y[1])]
    w1 = np.abs(solve_ivp(f, (0, 5), [1.0, 0.0], max_step=0.01).y[1]).max()
    w2 = np.abs(solve_ivp(f, (0, 10), [2.0, 0.0], max_step=0.01).y[1]).max()
    assert 2.6 < w1 < 2.8 and 4.4 < w2 < 4.6
    assert abs(damping(5.0) - 3.5) < 1e-12
