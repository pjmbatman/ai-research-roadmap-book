"""9.3 퀴즈(content/09-numerical-methods/03-numerical-linear-algebra/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import hilbert

QUIZ = Path(__file__).parents[3] / "content/09-numerical-methods/03-numerical-linear-algebra/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def laplacian_2d(m):
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m))
    return (sp.kron(sp.identity(m), T) + sp.kron(T, sp.identity(m))).tocsr()


def test_explicit_inverse():
    A = hilbert(11)
    x = np.ones(11)
    b = A @ x
    err = lambda xh: np.linalg.norm(xh - x) / np.linalg.norm(x)
    assert err(np.linalg.inv(A) @ b) > 10 * err(np.linalg.solve(A, b))
    # 희소 행렬의 역행렬은 꽉 찬다
    A = laplacian_2d(10).toarray()
    assert np.count_nonzero(np.abs(np.linalg.inv(A)) > 1e-14) == A.size
    assert correct("explicit-inverse") == [0]


def test_factor_once():
    n = 200
    A = rng.normal(size=(n, n)) + n * np.eye(n)
    B = rng.normal(size=(n, 5))
    lu = sla.lu_factor(A)
    X = np.column_stack([sla.lu_solve(lu, B[:, i]) for i in range(5)])
    assert np.allclose(A @ X, B)
    assert correct("factor-once") == [0]


def test_cg_requirements():
    A = laplacian_2d(30)
    b = np.ones(A.shape[0])
    op = spla.LinearOperator(A.shape, matvec=lambda v: A @ v)      # 행렬–벡터 곱만
    x, info = spla.cg(op, b, rtol=1e-10)
    assert info == 0 and np.allclose(A @ x, b, atol=1e-7)
    iters = []
    for m in [20, 40, 80]:                                          # cond ∝ m^2 → 반복 ∝ m
        it = []
        spla.cg(laplacian_2d(m), np.ones(m * m), rtol=1e-8, callback=lambda xk: it.append(1))
        iters.append(len(it))
    assert 1.6 < iters[1] / iters[0] < 2.4 and 1.6 < iters[2] / iters[1] < 2.4
    assert correct("cg-requirements") == [0, 1, 2]


def test_cholesky_jitter():
    x = np.linspace(0, 1, 50)
    K = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / 0.3**2)
    try:
        sla.cholesky(K, lower=True)
        failed = False
    except np.linalg.LinAlgError:
        failed = True
    assert failed
    sla.cholesky(K + 1e-6 * np.eye(50), lower=True)                  # jitter로 성공
    assert np.linalg.cond(K) > 1e16
    assert correct("cholesky-jitter") == [0]


def test_dense_memory():
    N = 100 * 100
    assert abs(N * N * 8 / 1e9 - Q["dense-memory"]["answer"]) <= Q["dense-memory"]["tolerance"]
    A = laplacian_2d(100)
    assert A.nnz == 49600 and (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) < 1e6


def test_sparse_fill_in():
    A = laplacian_2d(100).tocsc()
    nnz = {o: (lambda f: f.L.nnz + f.U.nnz)(spla.splu(A, permc_spec=o)) for o in ["NATURAL", "COLAMD"]}
    assert nnz["COLAMD"] > 6e5 and nnz["COLAMD"] > 10 * A.nnz
    assert nnz["NATURAL"] > 2 * nnz["COLAMD"]                         # 재정렬이 fill-in을 줄인다
    assert correct("sparse-fill-in") == [0]
