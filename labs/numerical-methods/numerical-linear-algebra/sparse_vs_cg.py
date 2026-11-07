"""9.3 실습: 대규모 희소 선형시스템(2·3차원 Poisson)을 직접법(희소 LU)과 CG로 풀어 시간·메모리·정확도를 비교한다.

- 직접법: scipy.sparse.linalg.splu (SuperLU, 기본 열 순서 COLAMD). 메모리는 L, U 인수의 저장량으로 잰다.
- 반복법: scipy.sparse.linalg.cg (전처리 없음). 메모리는 A(CSR)와 벡터 몇 개.
- 정확도: 해석적으로 아는 해 x_true에서 b = A x_true를 만들고 상대 오차를 잰다.

실행: OMP_NUM_THREADS=1 uv run --project labs python labs/numerical-methods/numerical-linear-algebra/sparse_vs_cg.py
시간은 컴퓨터마다 다르다. 크기에 따라 늘어나는 비율을 본다.
"""
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def laplacian(m, dim):
    """m^dim 격자의 2차 중심차분 Laplacian(Dirichlet 경계, 격자 간격 1). SPD."""
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m), format="csr")
    I = sp.identity(m, format="csr")
    if dim == 2:
        return (sp.kron(I, T) + sp.kron(T, I)).tocsr()
    return (sp.kron(sp.kron(I, I), T) + sp.kron(sp.kron(I, T), I) + sp.kron(sp.kron(T, I), I)).tocsr()


def csr_bytes(A):
    return A.data.nbytes + A.indices.nbytes + A.indptr.nbytes


def run(m, dim):
    A = laplacian(m, dim)
    N = A.shape[0]
    rng = np.random.default_rng(0)
    x_true = rng.normal(size=N)
    b = A @ x_true

    t0 = time.perf_counter()
    lu = spla.splu(A.tocsc())
    x_lu = lu.solve(b)
    t_lu = time.perf_counter() - t0
    lu_mb = (lu.L.nnz + lu.U.nnz) * 12 / 1e6          # 값 8바이트 + 인덱스 4바이트(대략)

    it = []
    t0 = time.perf_counter()
    x_cg, info = spla.cg(A, b, rtol=1e-10, maxiter=20_000, callback=lambda xk: it.append(1))
    t_cg = time.perf_counter() - t0

    err = lambda x: np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    print(f"{dim}D {m:>4}^{dim} {N:>8d} {csr_bytes(A) / 1e6:8.1f} {lu_mb:9.1f} {t_lu:8.2f} "
          f"{err(x_lu):9.1e} {len(it):6d} {t_cg:7.2f} {err(x_cg):9.1e}")


def main():
    print(f"{'문제':>3} {'격자':>7} {'N':>8} {'A[MB]':>8} {'LU[MB]':>9} {'LU[s]':>8} "
          f"{'LU 오차':>9} {'CG 반복':>6} {'CG[s]':>7} {'CG 오차':>9}")
    for m in [50, 100, 200, 400]:
        run(m, 2)
    for m in [20, 30, 40]:
        run(m, 3)


if __name__ == "__main__":
    main()
