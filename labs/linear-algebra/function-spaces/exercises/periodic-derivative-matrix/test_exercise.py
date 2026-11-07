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


central_diff_matrix = load().central_diff_matrix


def grid(N, L):
    return L * np.arange(N) / N


def max_err(N, L):
    x = grid(N, L)
    u = np.exp(np.sin(2 * np.pi * x / L))
    du = (2 * np.pi / L) * np.cos(2 * np.pi * x / L) * u
    return np.abs(np.asarray(central_diff_matrix(N, L)) @ u - du).max()


def test_shape_and_constant():
    D = np.asarray(central_diff_matrix(10, 1.0))
    assert D.shape == (10, 10), f"shape은 (N, N) = (10, 10)이어야 합니다. 현재 {D.shape}"
    assert np.allclose(D @ np.ones(10), 0.0), "상수 함수의 미분은 0이어야 합니다"


def test_antisymmetric():
    D = np.asarray(central_diff_matrix(12, 3.0))
    assert np.allclose(D.T, -D), "주기 중심 차분 행렬은 반대칭(D^T = -D)이어야 합니다"


def test_periodic_wraparound():
    N, L = 8, 2 * np.pi
    D = np.asarray(central_diff_matrix(N, L))
    h = L / N
    assert np.isclose(D[0, 1], 1 / (2 * h)) and np.isclose(D[0, N - 1], -1 / (2 * h)), \
        "0번 행은 u_1과 u_{N-1}(주기 경계로 u_{-1})을 써야 합니다"
    assert np.isclose(D[N - 1, 0], 1 / (2 * h)), "N-1번 행의 u_{i+1}은 u_0입니다"


@pytest.mark.parametrize("L", [2 * np.pi, 1.0, 5.0])
def test_sine_derivative(L):
    N = 64
    x = grid(N, L)
    du = np.asarray(central_diff_matrix(N, L)) @ np.sin(2 * np.pi * x / L)
    exact = (2 * np.pi / L) * np.cos(2 * np.pi * x / L)
    assert np.abs(du - exact).max() < 0.01 * (2 * np.pi / L), f"L = {L}에서 sin(2πx/L)의 미분이 맞지 않습니다. h = L/N을 썼는지 확인하세요"


def test_second_order_convergence():
    ratio = max_err(32, 2.0) / max_err(64, 2.0)
    assert 3.5 < ratio < 4.5, f"N을 두 배로 하면 오차가 약 1/4로 줄어야 합니다. 현재 비율 {ratio:.2f}"
