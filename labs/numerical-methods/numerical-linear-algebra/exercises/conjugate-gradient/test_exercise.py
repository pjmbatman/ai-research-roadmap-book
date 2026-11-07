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


conjugate_gradient = load().conjugate_gradient


def counted(A):
    calls = []

    def matvec(v):
        calls.append(1)
        return A @ v
    return matvec, calls


def random_spd(n, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = np.linspace(1.0, 50.0, n)                   # 조건수 50
    return (Q * lam) @ Q.T, rng.normal(size=n)


def test_solves_random_spd():
    A, b = random_spd(30, 0)
    matvec, _ = counted(A)
    x, k = conjugate_gradient(matvec, b, tol=1e-10)
    assert np.linalg.norm(A @ x - b) <= 1e-9 * np.linalg.norm(b), "상대 잔차가 tol 수준이어야 합니다"
    assert np.allclose(x, np.linalg.solve(A, b), rtol=1e-7, atol=1e-9)
    assert k <= 30, f"30 x 30 SPD 시스템은 약 30번 안에 풀려야 합니다. 현재 {k}번"


def test_one_matvec_per_iteration():
    A, b = random_spd(20, 1)
    matvec, calls = counted(A)
    _, k = conjugate_gradient(matvec, b, tol=1e-8)
    assert len(calls) == k, f"matvec을 반복마다 한 번 호출해야 합니다: 반복 {k}번, 호출 {len(calls)}번"


def test_few_distinct_eigenvalues():
    rng = np.random.default_rng(2)
    Q, _ = np.linalg.qr(rng.normal(size=(40, 40)))
    lam = np.repeat([1.0, 3.0, 10.0], [10, 15, 15])  # 서로 다른 고유값 3개
    A = (Q * lam) @ Q.T
    b = rng.normal(size=40)
    x, k = conjugate_gradient(counted(A)[0], b, tol=1e-10)
    assert k <= 4, f"서로 다른 고유값이 3개면 CG는 약 3번에 수렴합니다. 현재 {k}번"
    assert np.allclose(A @ x, b, atol=1e-8)


def test_matrix_free_poisson():
    n = 64
    def matvec(v):                                    # 1차원 Poisson을 행렬 없이
        out = 2.0 * v
        out[1:] -= v[:-1]
        out[:-1] -= v[1:]
        return out
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    b = np.random.default_rng(3).normal(size=n)
    x, k = conjugate_gradient(matvec, b, tol=1e-10, max_iter=500)
    assert np.allclose(A @ x, b, atol=1e-8)
    assert k <= n + 5


def test_zero_rhs_returns_immediately():
    x, k = conjugate_gradient(lambda v: 2.0 * v, np.zeros(5))
    assert k == 0 and np.allclose(x, 0)
