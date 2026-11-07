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


change_basis = load().change_basis


@pytest.mark.parametrize("n", [2, 3, 5])
def test_same_map(n):
    rng = np.random.default_rng(n)
    A = rng.normal(size=(n, n))
    S = rng.normal(size=(n, n)) + n * np.eye(n)              # 가역
    A_new = np.asarray(change_basis(A, S))
    assert A_new.shape == (n, n), f"shape은 (n, n)이어야 합니다. 현재 {A_new.shape}"
    for _ in range(5):
        c = rng.normal(size=n)                               # 새 좌표
        assert np.allclose(S @ (A_new @ c), A @ (S @ c)), "새 행렬로 계산한 결과를 옛 좌표로 옮기면 A @ (S @ c)와 같아야 합니다"


def test_legendre_example():
    T = np.diag([0.0, 1.0, 2.0])
    S = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.5]])
    expected = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    got = np.asarray(change_basis(T, S))
    assert np.allclose(got, expected), f"T((3x^2 - 1)/2) = 1 + 2 (3x^2 - 1)/2 이므로 셋째 열은 (1, 0, 2)입니다. 현재\n{got}\nS A S^-1 순서로 곱하지 않았는지 확인합니다"


def test_invariants():
    rng = np.random.default_rng(42)
    A = rng.normal(size=(4, 4))
    S = rng.normal(size=(4, 4)) + 4 * np.eye(4)
    A_new = np.asarray(change_basis(A, S))
    assert np.isclose(np.trace(A_new), np.trace(A)), "닮은 행렬의 trace는 같습니다"
    assert np.isclose(np.linalg.det(A_new), np.linalg.det(A)), "닮은 행렬의 행렬식은 같습니다"


def test_identity_basis():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(change_basis(A, np.eye(2)), A), "basis를 바꾸지 않으면 행렬도 그대로입니다"
