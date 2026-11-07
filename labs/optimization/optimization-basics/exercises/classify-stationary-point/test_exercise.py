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


classify = load().classify


def test_not_stationary():
    assert classify(np.array([0.1, 0.0]), np.eye(2)) == "not-stationary", "gradient가 0이 아니면 stationary point가 아닙니다"


@pytest.mark.parametrize("H,expected", [
    (np.diag([2.0, 1.0]), "min"),
    (np.diag([-2.0, -0.5]), "max"),
    (np.diag([2.0, -1.0]), "saddle"),
    (np.array([[2.0, 1.0], [1.0, 2.0]]), "min"),        # 고윳값 3, 1
])
def test_definite_cases(H, expected):
    assert classify(np.zeros(2), H) == expected, f"Hessian 고윳값 {np.linalg.eigvalsh(H)}이면 {expected}입니다"


def test_diagonal_entries_are_not_enough():
    H = np.array([[1.0, 2.0], [2.0, 1.0]])               # 고윳값 3, -1
    assert classify(np.zeros(2), H) == "saddle", "대각 성분이 모두 양수여도 고윳값에 음수가 있으면 안장점입니다"


@pytest.mark.parametrize("H", [np.zeros((1, 1)), np.diag([1.0, 0.0]), np.diag([0.0, -3.0])])
def test_inconclusive(H):
    assert classify(np.zeros(H.shape[0]), H) == "inconclusive", "0에 가까운 고윳값이 있고 나머지 부호가 한쪽뿐이면 판정을 보류합니다"


def test_saddle_with_zero_eigenvalue():
    assert classify(np.zeros(3), np.diag([1.0, 0.0, -1.0])) == "saddle", "음·양 고윳값이 함께 있으면 0 고윳값이 있어도 안장점입니다"


def test_double_well_points():
    # f(x, y) = x^4/4 - x^2/2 + y^2/2 의 stationary point: x = -1, 0, 1 (y = 0)
    hess = lambda x: np.diag([3 * x**2 - 1, 1.0])
    assert [classify(np.zeros(2), hess(x)) for x in (-1.0, 0.0, 1.0)] == ["min", "saddle", "min"]
