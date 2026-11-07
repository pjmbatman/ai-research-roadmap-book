"""25.2 퀴즈(content/25-scientific-discovery/02-equation-discovery/quiz.toml) 정답과 본문 수치를 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.special import ellipk

QUIZ = Path(__file__).parents[3] / "content/25-scientific-discovery/02-equation-discovery/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_finite_difference_noise_scales_with_dt():
    rng = np.random.default_rng(0)
    sd = []
    for dt in (0.01, 0.001):
        eta = 0.01 * rng.normal(size=200000)                  # 측정 잡음만 (신호 0)
        sd.append(np.std((eta[2:] - eta[:-2]) / (2 * dt)))
        assert np.isclose(sd[-1], 0.01 / (np.sqrt(2) * dt), rtol=0.02)   # 본문의 σ/(√2 Δt)
    assert np.isclose(sd[1] / sd[0], 10, rtol=0.03)
    assert correct("finite-difference-noise") == [0]


def test_pi_group_count_drag():
    D = np.array([[1, 1, -2], [1, -3, 0], [0, 1, -1], [0, 1, 0], [1, -1, -1]]).T   # F, rho, v, D, mu
    n, r = D.shape[1], np.linalg.matrix_rank(D)
    assert r == 3
    assert n - r == Q["pi-group-count"]["answer"] == 2
    assert np.allclose(D @ np.array([1, -1, -2, -2, 0]), 0)   # F/(rho v^2 D^2)
    assert np.allclose(D @ np.array([0, 1, 1, 1, -1]), 0)     # Re = rho v D / mu


def test_pendulum_mass_cannot_enter():
    D = np.array([[0, 0, 1], [0, 1, 0], [0, 1, -2], [1, 0, 0], [0, 0, 0]]).T     # T, L, g, m, theta0
    mass_row = D[0]
    assert np.count_nonzero(mass_row) == 1 and mass_row[3] == 1   # 질량 차원은 m뿐
    assert correct("pendulum-mass") == [0]


def test_pendulum_series_in_text():
    th = np.linspace(0.01, 0.3, 30)
    exact = 4 * ellipk(np.sin(th / 2) ** 2)
    series = 2 * np.pi * (1 + th**2 / 16 + 11 * th**4 / 3072)   # 본문의 급수
    assert np.max(np.abs(exact - series) / th**6) < 0.01        # 남는 오차는 θ^6 차수
    assert np.isclose(2 * np.pi / 16, 0.393, atol=5e-4)
    assert np.isclose(2 * np.pi * 11 / 3072, 0.022, atol=5e-4)
    # 본문 셀 4의 식 7.066 + 0.057 θ^4 - 0.784 cos θ 를 전개한 계수
    assert np.isclose(7.066 - 0.784, 6.282)
    assert np.isclose(0.784 / 2, 0.392)
    assert np.isclose(0.057 - 0.784 / 24, 0.024, atol=5e-4)


def test_lotka_volterra_conserved_quantity():
    a, b, c, d = 1.0, 0.5, 0.2, 0.6
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0.5, 5, 100), rng.uniform(0.5, 5, 100)
    dx, dy = a * x - b * x * y, c * x * y - d * y
    dH = (c - d / x) * dx + (b - a / y) * dy                   # H = c x - d ln x + b y - a ln y
    assert np.allclose(dH, 0)


def test_choice_keys():
    assert correct("library-missing-term") == [0]
    assert correct("pareto-knee") == [0]
    assert correct("law-evidence") == [0, 1, 2]
