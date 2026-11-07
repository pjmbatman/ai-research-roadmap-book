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


simulate_ou = load().simulate_ou


def test_shape_and_initial_value():
    x = simulate_ou(2.0, 1.0, 0.0, 0.5, 0.1, 30, 7, np.random.default_rng(0))
    assert np.shape(x) == (31, 7), f"shape은 (n_steps + 1, n_paths) = (31, 7)이어야 합니다. 현재 {np.shape(x)}"
    assert np.allclose(x[0], 2.0), "0번째 행은 초기값 x0이어야 합니다"


def test_deterministic_without_noise():
    theta, mu, x0, dt = 1.5, 1.0, 3.0, 0.1
    x = simulate_ou(x0, theta, mu, 0.0, dt, 20, 3, np.random.default_rng(0))
    k = np.arange(21)
    expected = mu + (x0 - mu) * (1 - theta * dt) ** k
    assert np.allclose(x[:, 0], expected), "sigma = 0이면 X_k = mu + (x0 - mu)(1 - theta dt)^k 입니다"


def test_one_step_noise_scales_with_sqrt_dt():
    dt, sigma = 0.01, 2.0
    x = simulate_ou(0.0, 1.0, 0.0, sigma, dt, 1, 200_000, np.random.default_rng(1))
    var1 = x[1].var()
    assert abs(var1 - sigma**2 * dt) < 0.002, (
        f"한 step 뒤 분산은 sigma^2 dt = {sigma**2 * dt}이어야 합니다. 현재 {var1:.4f} "
        "(잡음에 dt 대신 sqrt(dt)를 곱했는지 확인하세요)")


def test_mean_and_stationary_variance():
    theta, mu, sigma, dt, x0 = 2.0, 1.0, 1.0, 0.01, 4.0
    x = simulate_ou(x0, theta, mu, sigma, dt, 400, 4000, np.random.default_rng(2))
    t = 0.5
    assert abs(x[50].mean() - (mu + (x0 - mu) * np.exp(-theta * t))) < 0.03, "평균은 mu + (x0 - mu) e^{-theta t}로 이완해야 합니다"
    stat_var = sigma**2 / (theta * (2 - theta * dt))     # EM의 정상 분산, dt가 작으면 sigma^2 / (2 theta)
    assert abs(x[-1].var() - stat_var) < 0.02, f"t = 4에서 분산은 정상 분산 {stat_var:.3f}에 가까워야 합니다"
