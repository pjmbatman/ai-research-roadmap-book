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


filter_and_stress = load().filter_and_stress


def test_block_constant_gives_zero():
    u = np.repeat([1.0, -2.0, 0.5, 3.0], 4)                   # (16,) 블록마다 상수
    ubar, tau = filter_and_stress(u, 4)
    assert np.allclose(ubar, [1.0, -2.0, 0.5, 3.0]), f"블록 평균이 틀렸습니다: {ubar}"
    assert np.allclose(tau, 0), "블록 안에서 값이 일정하면 tau = 0이어야 합니다"


def test_linear_profile():
    K, dx, s = 8, 0.1, 3.0
    x = np.arange(32) * dx
    u = s * x                                                 # 모든 블록에서 기울기 s
    ubar, tau = filter_and_stress(u, K)
    assert ubar.shape == (4,)
    expected = s**2 * dx**2 * (K**2 - 1) / 24
    assert np.allclose(tau, expected), f"선형 장에서 tau = s²Δx²(K²−1)/24 = {expected:.4f}이어야 합니다. 현재 {tau}"


def test_realizability_nonnegative():
    rng = np.random.default_rng(0)
    u = 1e3 + rng.normal(size=4096) * 1e-4                    # 큰 평균 + 작은 변동: 반올림 오차가 커지기 쉬운 경우
    _, tau = filter_and_stress(u, 16)
    assert np.all(tau >= 0), f"tau는 블록 안 분산의 절반이므로 음수가 될 수 없습니다. 최솟값 {tau.min():.3e}"


def test_batch_input():
    rng = np.random.default_rng(1)
    U = rng.normal(size=(5, 64))
    ubar, tau = filter_and_stress(U, 8)
    assert ubar.shape == (5, 8) and tau.shape == (5, 8), f"(B, Nf // K) shape이어야 합니다. 현재 {ubar.shape}, {tau.shape}"
    ub0, t0 = filter_and_stress(U[2], 8)
    assert np.allclose(ubar[2], ub0) and np.allclose(tau[2], t0), "배치의 각 행은 따로 계산한 결과와 같아야 합니다"
