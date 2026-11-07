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


solution_structure = load().solution_structure

A_EX = np.array([[1.0, 2.0, 0.0, 1.0], [2.0, 4.0, 1.0, 3.0], [3.0, 6.0, 1.0, 4.0]])   # 본문의 예, rank 2


def test_invertible_square():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    assert solution_structure(A, [1.0, 2.0]) == "unique", "가역 정사각 행렬은 모든 b에 대해 해가 하나입니다"


def test_body_example():
    assert solution_structure(A_EX, A_EX @ [1.0, 0.0, 2.0, 0.0]) == "infinite", "b가 column space 안에 있고 null space가 2차원입니다"
    assert solution_structure(A_EX, [1.0, 0.0, 0.0]) == "none", "y = (1, 1, -1)에 대해 y @ b = 1 ≠ 0이므로 해가 없습니다"


def test_overdetermined():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(20, 3))
    assert solution_structure(A, A @ [1.0, -1.0, 2.0]) == "unique", "b가 column space 안에 있고 열이 독립이면 해는 하나입니다"
    assert solution_structure(A, rng.normal(size=20)) == "none", "무작위 b는 3차원 column space 밖에 있습니다"


def test_underdetermined():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 6))
    assert solution_structure(A, rng.normal(size=3)) == "infinite", "full row rank인 미결정 시스템은 해가 무수히 많습니다"


def test_zero_matrix():
    A = np.zeros((2, 2))
    assert solution_structure(A, [0.0, 0.0]) == "infinite", "A = 0, b = 0이면 모든 x가 해입니다"
    assert solution_structure(A, [1.0, 0.0]) == "none", "A = 0, b ≠ 0이면 해가 없습니다"


def test_tolerance_is_used():
    rng = np.random.default_rng(2)
    c1, c2 = rng.normal(size=30), rng.normal(size=30)
    A = np.column_stack([c1, c2, c1 + c2 + 1e-9 * rng.normal(size=30)])   # 잡음 수준에서만 독립
    b = A @ [1.0, 2.0, 3.0]
    assert solution_structure(A, b, tol=1e-12) == "unique", "허용오차가 아주 작으면 셋째 열도 독립으로 셉니다"
    assert solution_structure(A, b, tol=1e-6) == "infinite", "허용오차 1e-6에서는 rank가 2이므로 해가 무수히 많습니다. tol을 matrix_rank에 넘겼는지 확인합니다"
