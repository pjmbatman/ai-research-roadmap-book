import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial import polynomial as P

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


derivative_matrix = load().derivative_matrix


@pytest.mark.parametrize("n", [0, 1, 3, 6])
def test_shape(n):
    D = np.asarray(derivative_matrix(n))
    assert D.shape == (n + 1, n + 1), f"shape은 (n + 1, n + 1) = {(n + 1, n + 1)}이어야 합니다. 현재 {D.shape}"


@pytest.mark.parametrize("n", [1, 2, 4, 6])
def test_matches_polyder(n):
    rng = np.random.default_rng(n)
    D = np.asarray(derivative_matrix(n))
    for _ in range(5):
        c = rng.normal(size=n + 1)
        expected = np.append(P.polyder(c), 0.0)          # 길이를 n + 1로 맞춘다
        assert np.allclose(D @ c, expected), f"D @ c가 도함수의 계수와 다릅니다: {D @ c} != {expected}"


def test_example():
    D = np.asarray(derivative_matrix(3))
    assert np.allclose(D @ [1.0, 2.0, 3.0, 4.0], [2.0, 6.0, 12.0, 0.0]), "(1 + 2x + 3x^2 + 4x^3)' = 2 + 6x + 12x^2"


def test_constants_in_kernel():
    D = np.asarray(derivative_matrix(4))
    assert np.allclose(D @ np.eye(5)[0], 0), "상수 함수 1의 도함수는 0입니다"
    assert np.linalg.matrix_rank(D) == 4, "핵은 상수 함수뿐이므로 rank는 n이어야 합니다"


def test_nilpotent():
    n = 5
    D = np.asarray(derivative_matrix(n))
    assert np.allclose(np.linalg.matrix_power(D, n + 1), 0), "n + 1번 미분하면 0이 되어야 합니다"
    assert not np.allclose(np.linalg.matrix_power(D, n), 0), "n번 미분한 x^n은 n!이므로 D^n은 0이 아닙니다"
