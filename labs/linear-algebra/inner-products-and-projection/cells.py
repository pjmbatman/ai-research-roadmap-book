"""7.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/linear-algebra/inner-products-and-projection/cells.py
"""
# %% cell 1 — 벡터·행렬 노름과 cosine similarity
import numpy as np

x = np.array([3.0, -4.0, 0.0, 1.0])
print("L1 =", np.linalg.norm(x, 1), " L2 =", round(np.linalg.norm(x), 3), " Linf =", np.linalg.norm(x, np.inf))

M = np.array([[3.0, 0.0],
              [4.0, 5.0]])                                   # (2, 2)
print("Frobenius =", round(np.linalg.norm(M, "fro"), 3),
      " spectral =", round(np.linalg.norm(M, 2), 3),
      " nuclear =", round(np.linalg.norm(M, "nuc"), 3))

s = np.array([0.1, 0.8, 0.3, 0.05])                          # 스펙트럼 하나
t = 2.0 * s                                                  # 농도가 두 배인 같은 물질
cos = s @ t / (np.linalg.norm(s) * np.linalg.norm(t))
print("cosine =", round(cos, 6), " 유클리드 거리 =", round(np.linalg.norm(s - t), 3))

# %% cell 2 — 직교행렬은 길이와 각도를 보존한다
rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))                 # (4, 4) 직교행렬
u, v = rng.normal(size=4), rng.normal(size=4)
print("Q^T Q = I ?      ", np.allclose(Q.T @ Q, np.eye(4)))
print("|Qu|, |u|        ", round(np.linalg.norm(Q @ u), 6), round(np.linalg.norm(u), 6))
print("<Qu,Qv>, <u,v>   ", round((Q @ u) @ (Q @ v), 6), round(u @ v, 6))

# %% cell 3 — 사영행렬
A = np.array([[1.0, 0.0],
              [1.0, 1.0],
              [1.0, 2.0]])                                   # (3, 2), 열이 평면을 span
P = A @ np.linalg.solve(A.T @ A, A.T)                        # (3, 3) 사영행렬
b = np.array([6.0, 0.0, 0.0])
p = P @ b
print("P b       =", np.round(p, 4))
print("P^2 = P ? ", np.allclose(P @ P, P), "  P^T = P ?", np.allclose(P.T, P))
print("A^T (b-p) =", np.round(A.T @ (b - p), 12))

# %% cell 4 — 최소제곱: 용수철 상수 추정
rng = np.random.default_rng(1)
extension_m = np.linspace(0.01, 0.10, 12)                    # (12,) 늘어난 길이
force_n = 250.0 * extension_m + 0.3 + rng.normal(scale=0.4, size=12)
X = np.column_stack([extension_m, np.ones(12)])              # (12, 2)

w_ne = np.linalg.solve(X.T @ X, X.T @ force_n)               # 정규방정식
Qx, R = np.linalg.qr(X)                                      # X = Q R
w_qr = np.linalg.solve(R, Qx.T @ force_n)                    # R w = Q^T y
w_ls = np.linalg.lstsq(X, force_n)[0]
print("k [N/m], c [N]  정규방정식:", np.round(w_ne, 3), " QR:", np.round(w_qr, 3), " lstsq:", np.round(w_ls, 3))
r = force_n - X @ w_ls
print(f"max |X^T r| = {np.abs(X.T @ r).max():.1e}")

# %% cell 5 — pseudoinverse와 최소 노름 해
A1 = np.array([[1.0, 1.0]])                                  # (1, 2) 방정식 x1 + x2 = 2
b1 = np.array([2.0])
x_pinv = np.linalg.pinv(A1) @ b1
print("pinv 해 =", x_pinv, " 노름 =", round(np.linalg.norm(x_pinv), 4))
for x_other in [np.array([2.0, 0.0]), np.array([3.0, -1.0])]:
    print("다른 해 =", x_other, " 노름 =", round(np.linalg.norm(x_other), 4), " A x =", A1 @ x_other)
