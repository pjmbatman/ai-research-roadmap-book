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


conv_matrix = load().conv_matrix


def direct(kernel, x):
    T, c = len(x), (len(kernel) - 1) // 2
    return np.array([sum(kernel[j] * x[(t + j - c) % T] for j in range(len(kernel))) for t in range(T)])


@pytest.mark.parametrize("k,T", [(1, 4), (3, 6), (5, 9), (7, 7)])
def test_matches_definition(k, T):
    rng = np.random.default_rng(10 * k + T)
    kernel, x = rng.normal(size=k), rng.normal(size=T)
    M = conv_matrix(kernel, T)
    assert np.shape(M) == (T, T), f"shape은 (T, T) = ({T}, {T})여야 합니다. 현재 {np.shape(M)}"
    assert np.allclose(M @ x, direct(kernel, x)), "M @ x가 정의식의 합성곱과 다릅니다"


def test_center_tap_on_diagonal():
    M = conv_matrix(np.array([0.0, 1.0, 0.0]), 5)
    assert np.allclose(M, np.eye(5)), "가운데 원소만 1인 kernel은 항등 행렬이어야 합니다(가운데 원소가 x_t에 곱해짐)"


def test_wraps_around():
    M = conv_matrix(np.array([1.0, 0.0, 0.0]), 5)
    assert M[0, 4] == 1.0, "행 0에서 kernel의 첫 원소는 x_{-1} = x_{T-1} 열(4)에 놓여야 합니다(원형 경계)"


def test_shift_equivariance():
    rng = np.random.default_rng(1)
    T = 8
    M = conv_matrix(rng.normal(size=3), T)
    S = np.roll(np.eye(T), 1, axis=0)
    assert np.allclose(M @ S, S @ M), "원형 합성곱 행렬은 이동 행렬과 교환되어야 합니다(이동 등변성)"


def test_weight_sharing():
    kernel = np.array([0.5, -1.0, 2.0])
    M = conv_matrix(kernel, 6)
    for t in range(1, 6):
        assert np.allclose(M[t], np.roll(M[t - 1], 1)), "각 행은 앞 행을 한 칸 원형 이동한 것이어야 합니다"
    assert set(np.unique(M)) <= {0.0, 0.5, -1.0, 2.0}
