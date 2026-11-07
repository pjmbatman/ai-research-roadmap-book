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


min_norm_solution = load().min_norm_solution


def null_basis(A):
    """테스트용: SVD로 null space의 정규직교 basis를 구한다(7.6장)."""
    _, s, Vt = np.linalg.svd(A)
    r = np.sum(s > 1e-10)
    return Vt[r:].T                                  # (n, n - r)


def test_line_example():
    x = np.asarray(min_norm_solution(np.array([[1.0, 1.0]]), np.array([2.0])))
    assert np.allclose(x, [1.0, 1.0]), f"x1 + x2 = 2의 최소 노름 해는 (1, 1)입니다. 현재 {x}"


@pytest.mark.parametrize("m,n", [(1, 3), (2, 5), (4, 10)])
def test_solves_and_matches_pinv(m, n):
    rng = np.random.default_rng(m * 10 + n)
    A, b = rng.normal(size=(m, n)), rng.normal(size=m)
    x = np.asarray(min_norm_solution(A, b))
    assert x.shape == (n,), f"x의 shape은 (n,) = ({n},)이어야 합니다. 현재 {x.shape}"
    assert np.allclose(A @ x, b), "A x = b를 만족해야 합니다"
    assert np.allclose(x, np.linalg.pinv(A) @ b), "pseudoinverse 해와 같아야 합니다"


def test_orthogonal_to_null_space():
    rng = np.random.default_rng(1)
    A, b = rng.normal(size=(3, 7)), rng.normal(size=3)
    x = np.asarray(min_norm_solution(A, b))
    N = null_basis(A)                                # (7, 4)
    assert np.allclose(N.T @ x, 0), "최소 노름 해는 null space와 직교합니다(row space 안에 있음)"


def test_smaller_than_other_solutions():
    rng = np.random.default_rng(2)
    A, b = rng.normal(size=(2, 6)), rng.normal(size=2)
    x = np.asarray(min_norm_solution(A, b))
    N = null_basis(A)
    for _ in range(10):
        other = x + N @ rng.normal(size=N.shape[1])
        assert np.allclose(A @ other, b)
        assert np.linalg.norm(x) < np.linalg.norm(other), "null space 성분을 더한 다른 해보다 노름이 작아야 합니다"
