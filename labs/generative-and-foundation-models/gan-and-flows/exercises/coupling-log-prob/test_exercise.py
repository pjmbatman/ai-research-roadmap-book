import importlib.util
import os
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


flow_log_prob = load().flow_log_prob


def test_identity_flow_is_standard_normal():
    x = np.array([[0.0, 0.0], [1.0, -2.0]])
    got = flow_log_prob(x, lambda u: np.zeros_like(u), lambda u: np.zeros_like(u))
    assert np.allclose(got, multivariate_normal(np.zeros(2), np.eye(2)).logpdf(x)), "s = t = 0이면 표준정규분포입니다"


def test_constant_scale_and_shift():
    # s = log 3, t = 1: x2 = 3 z2 + 1 ~ N(1, 9), x1 ~ N(0, 1)
    x = np.array([[0.5, 1.0], [-1.0, 4.0], [2.0, -3.0]])
    got = flow_log_prob(x, lambda u: np.full_like(u, np.log(3.0)), lambda u: np.ones_like(u))
    expect = multivariate_normal([0.0, 1.0], np.diag([1.0, 9.0])).logpdf(x)
    assert np.allclose(got, expect), f"x2 ~ N(1, 3^2)의 밀도와 달라요. log-det의 부호를 확인하세요. 현재 {np.round(got, 3)}"


def s_fn(u):
    return 0.7 * np.tanh(u)


def t_fn(u):
    return 0.8 * u**2 - 1.0


def test_total_mass_is_one():
    g = np.linspace(-9, 14, 921)
    X = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    mass = np.exp(flow_log_prob(X, s_fn, t_fn)).sum() * (g[1] - g[0]) ** 2
    assert abs(mass - 1.0) < 2e-3, f"격자 적분한 전체 질량이 1이어야 합니다. 현재 {mass:.4f}"


def test_matches_samples():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(200_000, 2))
    x = np.stack([z[:, 0], z[:, 1] * np.exp(s_fn(z[:, 0])) + t_fn(z[:, 0])], 1)
    # 상자 [0.5, 1.5] x [-1, 1]의 확률: 표본 비율 vs 밀도의 격자 적분
    frac = np.mean((x[:, 0] > 0.5) & (x[:, 0] < 1.5) & (x[:, 1] > -1) & (x[:, 1] < 1))
    g1, g2 = np.linspace(0.5, 1.5, 201), np.linspace(-1, 1, 401)
    G = np.stack(np.meshgrid(g1, g2), -1).reshape(-1, 2)
    integral = np.exp(flow_log_prob(G, s_fn, t_fn)).reshape(len(g2), len(g1))
    integral = np.trapezoid(np.trapezoid(integral, dx=g1[1] - g1[0], axis=1), dx=g2[1] - g2[0])
    assert abs(frac - integral) < 0.003, f"표본 비율 {frac:.4f}, 밀도 적분 {integral:.4f}"
