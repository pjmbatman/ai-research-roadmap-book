import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load()
ssm_scan, ssm_kernel, ssm_conv = mod.ssm_scan, mod.ssm_kernel, mod.ssm_conv


def random_stable(N, seed):
    """스펙트럼 반지름이 0.95 이하인 무작위 이산 SSM"""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N, N))
    A *= 0.95 / max(abs(np.linalg.eigvals(A)))
    return A, rng.normal(size=(N, 1)), rng.normal(size=(1, N)), rng.normal(size=40)


@pytest.mark.parametrize("N,seed", [(1, 0), (2, 1), (4, 2), (8, 3)])
def test_scan_equals_conv(N, seed):
    A, B, C, u = random_stable(N, seed)
    y1, y2 = ssm_scan(A, B, C, u), ssm_conv(A, B, C, u)
    assert np.shape(y1) == (40,) and np.shape(y2) == (40,), "출력 shape은 (L,)이어야 합니다"
    assert np.allclose(y1, y2), "재귀와 합성곱의 출력이 달라야 할 이유가 없습니다"


def test_scan_known_scalar():
    y = ssm_scan(np.array([[0.5]]), np.array([[1.0]]), np.array([[2.0]]), np.array([1.0, 0.0, 0.0, 1.0]))
    assert np.allclose(y, [2.0, 1.0, 0.5, 2.25]), "x_k = 0.5 x_{k-1} + u_k, y = 2 x_k를 손으로 계산한 값과 다릅니다"


def test_kernel_is_impulse_response():
    A, B, C, _ = random_stable(3, 7)
    impulse = np.zeros(25)
    impulse[0] = 1.0
    assert np.allclose(ssm_kernel(A, B, C, 25), ssm_scan(A, B, C, impulse)), "kernel은 단위 충격에 대한 응답이어야 합니다"


def test_kernel_first_terms():
    A, B, C, _ = random_stable(3, 5)
    K = ssm_kernel(A, B, C, 3)
    assert np.isclose(K[0], (C @ B).item()), "K_0 = C B̄ 입니다"
    assert np.isclose(K[2], (C @ A @ A @ B).item()), "K_2 = C Ā² B̄ 입니다"


def test_causal():
    A, B, C, u = random_stable(4, 9)
    u2 = u.copy()
    u2[20:] += 5.0
    for f in (ssm_scan, ssm_conv):
        assert np.allclose(f(A, B, C, u)[:20], f(A, B, C, u2)[:20]), "y_k는 k 이후의 입력에 의존하면 안 됩니다"
