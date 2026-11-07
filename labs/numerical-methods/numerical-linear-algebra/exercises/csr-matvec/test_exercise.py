import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


csr_matvec = load().csr_matvec


def test_example_from_problem():
    data, indices, indptr = [4.0, 1.0, 2.0, 3.0], [0, 2, 0, 1], [0, 2, 2, 4]
    y = csr_matvec(data, indices, indptr, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(y, [7.0, 0.0, 8.0]), f"문제의 예에서 A @ [1, 2, 3] = [7, 0, 8]입니다. 현재 {y}"


@pytest.mark.parametrize("m,n,density", [(5, 5, 0.4), (30, 20, 0.1), (100, 300, 0.02)])
def test_matches_scipy(m, n, density):
    A = sp.random(m, n, density=density, format="csr", random_state=m + n)
    x = np.random.default_rng(m).normal(size=n)
    y = csr_matvec(A.data, A.indices, A.indptr, x)
    assert np.shape(y) == (m,), f"반환 shape은 ({m},)이어야 합니다. 현재 {np.shape(y)}"
    assert np.allclose(y, A @ x)


def test_empty_rows_and_rectangular():
    A = sp.csr_matrix(np.array([[0.0, 0.0], [1.0, -2.0], [0.0, 0.0], [0.0, 5.0]]))   # (4, 2)
    y = csr_matvec(A.data, A.indices, A.indptr, np.array([3.0, 1.0]))
    assert np.allclose(y, [0.0, 1.0, 0.0, 5.0]), "빈 행의 결과는 0입니다"


def test_poisson_stencil():
    m = 30
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m))
    A = (sp.kron(sp.identity(m), T) + sp.kron(T, sp.identity(m))).tocsr()   # (900, 900)
    x = np.random.default_rng(0).normal(size=m * m)
    assert np.allclose(csr_matvec(A.data, A.indices, A.indptr, x), A @ x)
