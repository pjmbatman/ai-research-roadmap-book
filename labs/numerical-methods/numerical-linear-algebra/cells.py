"""9.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/numerical-methods/numerical-linear-algebra/cells.py
"""
# %% cell 1 — LU 분해와 피벗팅
import numpy as np
import scipy.linalg as sla

A = np.array([[1e-20, 1.0],
              [1.0,   1.0]])
b = np.array([1.0, 2.0])                       # 참해 ≈ (1, 1)

# 피벗팅 없는 Gauss 소거: 1e-20을 pivot으로 쓴다
l21 = A[1, 0] / A[0, 0]                        # 1e20
u22 = A[1, 1] - l21 * A[0, 1]                  # 1 - 1e20 → -1e20 (1이 사라진다)
x2 = (b[1] - l21 * b[0]) / u22
x1 = (b[0] - A[0, 1] * x2) / A[0, 0]
print("피벗팅 없음 :", np.array([x1, x2]))

lu, piv = sla.lu_factor(A)                     # 부분 피벗팅: 큰 원소를 pivot으로
print("부분 피벗팅 :", sla.lu_solve((lu, piv), b))
print("cond(A)     :", f"{np.linalg.cond(A):.2f}")

# %% cell 2 — Cholesky: 양의 정부호 검사와 jitter
x = np.linspace(0, 1, 50)                                  # (50,) 입력 위치
K = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / 0.3**2)  # (50, 50) RBF 커널
print(f"cond(K) = {np.linalg.cond(K):.1e}, 최소 고유값 = {np.linalg.eigvalsh(K)[0]:.1e}")
try:
    sla.cholesky(K, lower=True)
except np.linalg.LinAlgError:
    print("jitter 없음 : LinAlgError (수치적으로 양의 정부호가 아니다)")

L = sla.cholesky(K + 1e-6 * np.eye(50), lower=True)        # K + jitter*I
y = np.sin(2 * np.pi * x)                                  # (50,) 관측값
alpha = sla.cho_solve((L, True), y)                        # (K + jitter I)^-1 y
print("jitter 1e-6 : 성공, 잔차 =",
      f"{np.linalg.norm((K + 1e-6 * np.eye(50)) @ alpha - y):.1e}")

# %% cell 3 — 반복법: Jacobi · Gauss–Seidel · CG
import scipy.sparse.linalg as spla

n = 50
A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)       # (n, n) 1차원 Poisson
b = np.ones(n)

def stationary(A, b, M, tol=1e-8, max_iter=50_000):
    """x_{k+1} = x_k + M^{-1}(b - A x_k). M = 대각(Jacobi) 또는 하삼각(Gauss–Seidel)."""
    x = np.zeros_like(b)
    for k in range(max_iter):
        r = b - A @ x
        if np.linalg.norm(r) <= tol * np.linalg.norm(b):
            return k
        x = x + sla.solve_triangular(M, r, lower=True)
    return max_iter

print("Jacobi       반복 수:", stationary(A, b, np.diag(np.diag(A))))
print("Gauss–Seidel 반복 수:", stationary(A, b, np.tril(A)))
it = []
x_cg, info = spla.cg(A, b, rtol=1e-8, callback=lambda xk: it.append(1))
print("CG           반복 수:", len(it))

# %% cell 4 — 희소 행렬: 2차원 Poisson, CSR, fill-in
import scipy.sparse as sp

m = 100                                                     # m x m 격자 → N = 10^4 미지수
T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m))
I = sp.identity(m)
A = (sp.kron(I, T) + sp.kron(T, I)).tocsr()                 # (N, N) 5점 Laplacian
N = A.shape[0]
csr_mb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e6
print(f"N = {N}, nnz = {A.nnz}, 밀도 = {A.nnz / N**2:.1e}")
print(f"저장: CSR {csr_mb:.2f} MB, dense {N * N * 8 / 1e9:.1f} GB")

b = np.ones(N)
for order in ["NATURAL", "COLAMD"]:
    lu = spla.splu(A.tocsc(), permc_spec=order)
    print(f"LU({order:7s}) nnz(L+U) = {lu.L.nnz + lu.U.nnz:7d} (A의 {(lu.L.nnz + lu.U.nnz) / A.nnz:.0f}배)")
x_direct = lu.solve(b)

it = []
x_cg, info = spla.cg(A, b, rtol=1e-10, callback=lambda xk: it.append(1))
print(f"CG 반복 수 = {len(it)}, 직접법과의 상대 차이 = "
      f"{np.linalg.norm(x_cg - x_direct) / np.linalg.norm(x_direct):.1e}")

# %% cell 5 — 행렬을 만들지 않는 계산과 무작위 SVD
def apply_laplacian(v):                                     # A @ v를 행렬 없이: 5점 stencil
    u = v.reshape(m, m)                                     # (m, m) 격자 값
    out = 4.0 * u
    out[1:, :] -= u[:-1, :]
    out[:-1, :] -= u[1:, :]
    out[:, 1:] -= u[:, :-1]
    out[:, :-1] -= u[:, 1:]
    return out.ravel()                                      # (N,)

op = spla.LinearOperator((N, N), matvec=apply_laplacian)
x_mf, info = spla.cg(op, b, rtol=1e-10)
print(f"matrix-free CG와 직접법의 상대 차이 = {np.linalg.norm(x_mf - x_direct) / np.linalg.norm(x_direct):.1e}")

rng = np.random.default_rng(0)
U0, _ = np.linalg.qr(rng.normal(size=(2000, 30)))           # (2000, 30)
V0, _ = np.linalg.qr(rng.normal(size=(500, 30)))            # (500, 30)
M = (U0 * 0.7 ** np.arange(30)) @ V0.T                      # (2000, 500) 특잇값이 빠르게 감소
k, p = 10, 10                                               # 원하는 rank, 여유 표본
Q, _ = np.linalg.qr(M @ rng.normal(size=(500, k + p)))      # (2000, k+p) 열공간의 근사 기저
s_rand = np.linalg.svd(Q.T @ M, compute_uv=False)[:k]       # (k+p, 500) 작은 행렬의 SVD
s_full = np.linalg.svd(M, compute_uv=False)[:k]
print(f"무작위 SVD 상위 {k}개 특잇값의 최대 상대 오차 = {np.max(np.abs(s_rand - s_full) / s_full):.1e}")
