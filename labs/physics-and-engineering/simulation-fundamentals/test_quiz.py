"""26.1 퀴즈(content/26-physics-and-engineering/01-simulation-fundamentals/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/26-physics-and-engineering/01-simulation-fundamentals/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def shock_front(conservative, N=200, steps=200):
    dx = 2.0 / N
    x = (np.arange(N) + 0.5) * dx
    u = np.where(x < 0.5, 1.0, 0.0)
    dt = 0.4 * dx
    for _ in range(steps):
        if conservative:
            F = 0.5 * u**2
            u = u - dt / dx * (F - np.concatenate([[0.5], F[:-1]]))
        else:
            u = u - dt / dx * u * (u - np.concatenate([[1.0], u[:-1]]))
    return x[np.argmax(u < 0.5)], 0.5 + 0.5 * dt * steps


def test_conservative_shock_speed():
    for N, steps in [(200, 200), (800, 800)]:                 # 격자를 줄여도
        front_c, exact = shock_front(True, N, steps)
        front_n, _ = shock_front(False, N, steps)
        assert abs(front_c - exact) < 0.03
        assert abs(front_n - 0.5) < 0.02                       # 비보존형은 출발점에 머문다
    assert correct("conservative-shock-speed") == [0]


def test_pipe_reynolds_number():
    assert np.isclose(0.5 * 0.02 / 1.0e-6, Q["pipe-reynolds-number"]["answer"])


def test_same_reynolds_similarity():
    base = 1.0 * 1.0 / 0.01
    opts = [(2, 0.5, 0.01), (2, 1, 0.02), (2, 1, 0.01), (1, 2, 0.005)]
    expected = [i for i, (U, L, nu) in enumerate(opts) if np.isclose(U * L / nu, base)]
    assert correct("same-reynolds-similarity") == expected == [0, 1]


def steady_central(Pe_cell, N=40):
    A = (np.diag(np.full(N - 1, 2.0)) + np.diag(np.full(N - 2, -(1 + Pe_cell / 2)), -1)
         + np.diag(np.full(N - 2, -(1 - Pe_cell / 2)), 1))
    b = np.zeros(N - 1)
    b[-1] = 1 - Pe_cell / 2
    return np.linalg.solve(A, b)


def test_max_grid_spacing_peclet():
    U, D = 0.2, 1.0e-3
    dx_mm = Q["max-grid-spacing-peclet"]["answer"]
    assert np.isclose(2 * D / U * 1000, dx_mm)
    assert steady_central(U * (dx_mm * 0.9e-3) / D).min() > -1e-12     # 한계 안쪽: 진동 없음
    assert steady_central(U * (dx_mm * 1.5e-3) / D).min() < -1e-3      # 한계 밖: 음수 농도


def test_choice_keys():
    assert correct("unconverged-training-data") == [0]
    assert correct("verification-before-validation") == [0]
