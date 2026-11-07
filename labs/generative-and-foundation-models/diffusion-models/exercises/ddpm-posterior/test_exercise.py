import importlib.util
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


posterior = load().posterior
BETAS = np.linspace(1e-4, 0.02, 1000)
ABAR = np.cumprod(1 - BETAS)


def bayes(x0, xt, t):
    """두 가우시안 인자의 곱을 정밀도로 직접 계산한다."""
    a_prev = ABAR[t - 2]
    b = BETAS[t - 1]
    prec = 1 / (1 - a_prev) + (1 - b) / b                       # q(x_{t-1}|x0)와 q(x_t|x_{t-1})의 정밀도
    mean = (np.sqrt(a_prev) * x0 / (1 - a_prev) + np.sqrt(1 - b) * xt / b) / prec
    return mean, 1 / prec


def test_matches_bayes_rule():
    rng = np.random.default_rng(0)
    for t in [2, 10, 300, 1000]:
        x0, xt = rng.normal(size=5), rng.normal(size=5)
        m, v = posterior(x0, xt, t, BETAS)
        m_ref, v_ref = bayes(x0, xt, t)
        assert np.allclose(m, m_ref) and np.allclose(v, v_ref), f"t={t}에서 사후 평균·분산이 베이즈 정리의 결과와 다릅니다"


def test_t_equals_one():
    m, v = posterior(np.array([0.7]), np.array([0.69]), 1, BETAS)
    assert np.allclose(m, [0.7]) and np.allclose(v, [0.0]), "t = 1이면 평균은 x0, 분산은 0입니다(abar_0 = 1)"


def test_variance_shape_and_bound():
    m, v = posterior(np.zeros(3), np.ones(3), 500, BETAS)
    assert np.shape(v) == (3,), "var는 mean과 같은 shape으로 반환합니다"
    assert np.all(v > 0) and np.all(v < BETAS[499]), "사후분산은 0보다 크고 beta_t보다 작습니다"


def test_monte_carlo():
    rng = np.random.default_rng(1)
    t, x0 = 200, 1.3
    n = 400_000
    x_prev = np.sqrt(ABAR[t - 2]) * x0 + np.sqrt(1 - ABAR[t - 2]) * rng.normal(size=n)
    x_t = np.sqrt(1 - BETAS[t - 1]) * x_prev + np.sqrt(BETAS[t - 1]) * rng.normal(size=n)
    sel = np.abs(x_t - 0.5) < 0.01                              # x_t ≈ 0.5인 경로만 고른다
    m, v = posterior(x0, 0.5, t, BETAS)
    assert abs(x_prev[sel].mean() - float(m)) < 0.01, f"시뮬레이션 평균 {x_prev[sel].mean():.4f}, 공식 {float(m):.4f}"
    assert abs(x_prev[sel].var() / float(v) - 1) < 0.15, "시뮬레이션 분산과 공식이 다릅니다"
