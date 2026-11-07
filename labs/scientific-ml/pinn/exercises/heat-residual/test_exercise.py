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


heat_residual = load().heat_residual
ALPHA = 0.1


def grid(nx, nt, alpha=ALPHA):
    x, t = np.linspace(0, 1, nx), np.linspace(0, 1, nt)
    u = np.exp(-alpha * np.pi**2 * t)[:, None] * np.sin(np.pi * x)[None, :]   # (nt, nx)
    return u, x[1] - x[0], t[1] - t[0]


def test_shape():
    u, dx, dt = grid(11, 21)
    r = heat_residual(u, dx, dt, ALPHA)
    assert np.shape(r) == (19, 9), f"shape은 (Nt-2, Nx-2) = (19, 9)여야 합니다. 현재 {np.shape(r)}"


def test_exact_solution_small_and_second_order():
    u1, dx1, dt1 = grid(21, 41)
    u2, dx2, dt2 = grid(41, 81)
    r1 = np.abs(heat_residual(u1, dx1, dt1, ALPHA)).max()
    r2 = np.abs(heat_residual(u2, dx2, dt2, ALPHA)).max()
    assert r1 < 1e-2, f"정확해의 잔차가 너무 큽니다(각 항의 크기는 약 1): {r1:.2e}"
    assert 3.5 < r1 / r2 < 4.5, f"간격을 절반으로 하면 잔차가 약 1/4로 줄어야 합니다(2차). 비율 {r1 / r2:.2f}"


def test_wrong_diffusivity_detected():
    u, dx, dt = grid(41, 81, alpha=0.2)                    # α = 0.2로 만든 해를 α = 0.1로 검사
    r = np.abs(heat_residual(u, dx, dt, ALPHA)).max()
    assert r > 0.1, f"확산 계수가 틀린 해는 잔차가 커야 합니다. 현재 {r:.2e}"


def test_linear_steady_state_zero():
    x = np.linspace(0, 1, 9)
    u = np.tile(x, (5, 1))                                 # 모든 시각에서 u = x, (5, 9)
    r = heat_residual(u, x[1] - x[0], 0.1, ALPHA)
    assert np.allclose(r, 0, atol=1e-12), "시간에 무관한 선형 함수는 열방정식을 정확히 만족합니다"
