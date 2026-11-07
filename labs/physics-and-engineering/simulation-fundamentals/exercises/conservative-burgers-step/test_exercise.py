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


burgers_fv_step = load().burgers_fv_step


def square_wave(N=200):
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx                         # (N,) 셀 중심
    return np.where((x > 0.2) & (x < 0.5), 1.0, 0.0), x, dx


def test_constant_state_unchanged():
    u = np.full(50, 0.7)
    new = burgers_fv_step(u, 0.02, 0.01)
    assert np.allclose(new, 0.7), "상수 상태에서는 모든 면 유속이 같아 값이 변하지 않아야 합니다"


def test_input_not_modified():
    u, _, dx = square_wave()
    before = u.copy()
    burgers_fv_step(u, dx, 0.5 * dx)
    assert np.array_equal(u, before), "입력 배열을 제자리에서 바꾸지 마세요"


def test_mass_conserved():
    rng = np.random.default_rng(0)
    u = rng.normal(size=128)
    dx = 1.0 / 128
    dt = 0.4 * dx / np.abs(u).max()
    mass0 = u.sum() * dx
    for _ in range(50):
        u = burgers_fv_step(u, dx, dt)
    assert abs(u.sum() * dx - mass0) < 1e-12, "주기 경계의 보존형 이산화는 전체 질량을 정확히 보존해야 합니다"


def test_no_new_extrema():
    u, _, dx = square_wave()
    for _ in range(100):
        u = burgers_fv_step(u, dx, 0.5 * dx)
    assert u.min() > -1e-12 and u.max() < 1 + 1e-12, f"새 극값이 생겼습니다: [{u.min():.3f}, {u.max():.3f}]"


def test_shock_speed():
    u, x, dx = square_wave(400)
    dt, steps = 0.4 * dx, 500                             # t = 0.5
    for _ in range(steps):
        u = burgers_fv_step(u, dx, dt)
    t = dt * steps
    right = x[(x > 0.5) & (u < 0.5)][0]                   # 충격파는 x = 0.5에서 출발, 속도 (1 + 0) / 2
    assert abs(right - (0.5 + 0.5 * t)) < 3 * dx, f"충격파 위치 {right:.3f}, 이론값 {0.5 + 0.5 * t:.3f}"
