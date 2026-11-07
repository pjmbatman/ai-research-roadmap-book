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


mod = load()
cfm_pair, velocity_from_eps = mod.cfm_pair, mod.velocity_from_eps
rng = np.random.default_rng(0)


def test_shapes_and_endpoints():
    x0, x1 = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    xt, u = cfm_pair(x0, x1, rng.uniform(size=5))
    assert np.shape(xt) == (5, 3) and np.shape(u) == (5, 3), f"shape은 (N, d) = (5, 3)이어야 합니다. 현재 {np.shape(xt)}, {np.shape(u)}"
    xt0, _ = cfm_pair(x0, x1, np.zeros(5))
    xt1, _ = cfm_pair(x0, x1, np.ones(5))
    assert np.allclose(xt0, x0), "t = 0이면 x_t는 잡음 x0여야 합니다"
    assert np.allclose(xt1, x1), "t = 1이면 x_t는 데이터 x1이어야 합니다"


def test_per_sample_time():
    x0, x1 = np.zeros((3, 2)), np.ones((3, 2))
    xt, _ = cfm_pair(x0, x1, np.array([0.0, 0.5, 1.0]))
    assert np.allclose(xt, [[0, 0], [0.5, 0.5], [1, 1]]), "표본 i는 자기 시간 t[i]를 써야 합니다 (t[:, None])"


def test_target_constant_along_path():
    x0, x1 = rng.normal(size=(4, 2)), rng.normal(size=(4, 2))
    _, u_a = cfm_pair(x0, x1, np.full(4, 0.1))
    _, u_b = cfm_pair(x0, x1, np.full(4, 0.9))
    assert np.allclose(u_a, u_b), "직선 경로의 목표 속도는 시간에 따라 바뀌지 않습니다"
    assert np.allclose(u_a, x1 - x0), "목표 속도는 x1 - x0입니다"


def test_delta_data_velocity_reaches_point():
    c = np.array([2.0, -1.0])
    t = np.array([0.2, 0.5, 0.8])
    xt = rng.normal(size=(3, 2))
    eps_hat = (xt - t[:, None] * c) / (1 - t[:, None])          # 데이터가 c 한 점일 때의 참 ε-예측
    v = velocity_from_eps(xt, eps_hat, t)
    assert np.allclose(v, (c - xt) / (1 - t[:, None])), "남은 시간 1 - t 안에 c에 도착하는 속도여야 합니다"


def test_true_noise_gives_pair_target():
    x0, x1, t = rng.normal(size=(6, 2)), rng.normal(size=(6, 2)), rng.uniform(0.1, 1.0, size=6)
    xt, u = cfm_pair(x0, x1, t)
    assert np.allclose(velocity_from_eps(xt, x0, t), u), "진짜 잡음 x0를 넣으면 그 짝의 목표 x1 - x0가 나와야 합니다"
