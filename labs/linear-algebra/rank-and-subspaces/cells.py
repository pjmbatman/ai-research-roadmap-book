"""7.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/linear-algebra/rank-and-subspaces/cells.py
"""
# %% cell 1 — 네 부분공간
import numpy as np

A = np.array([[1.0, 2.0, 0.0, 1.0],
              [2.0, 4.0, 1.0, 3.0],
              [3.0, 6.0, 1.0, 4.0]])          # (3, 4), 셋째 행 = 첫째 행 + 둘째 행
N = np.array([[-2.0, 1.0, 0.0, 0.0],
              [-1.0, 0.0, -1.0, 1.0]]).T     # (4, 2) null space의 basis
y = np.array([1.0, 1.0, -1.0])               # (3,) left null space의 basis

r = np.linalg.matrix_rank(A)
print("rank =", r, "  rank(A^T) =", np.linalg.matrix_rank(A.T))
print("dim: C(A) =", r, " C(A^T) =", r, " N(A) =", A.shape[1] - r, " N(A^T) =", A.shape[0] - r)
print("A @ N =\n", A @ N)                    # 각 행(row space)이 null space와 직교
print("y @ A =", y @ A)                      # 각 열(column space)이 y와 직교

# %% cell 2 — 해의 존재와 유일성
x0 = np.array([1.0, 0.0, 2.0, 0.0])
b1 = A @ x0                                  # column space 안의 b
b2 = np.array([1.0, 0.0, 0.0])               # column space 밖의 b
print("y @ b1 =", y @ b1, "  y @ b2 =", y @ b2)

x1 = x0 + 3.0 * N[:, 0] - 1.0 * N[:, 1]      # 특수해 + null space 성분
print("A @ x0 =", A @ x0, "  A @ x1 =", A @ x1)

x_ls = np.linalg.lstsq(A, b2)[0]             # 가장 가까운 근사해(7.4장)
print("b2의 최소제곱 잔차 =", np.round(b2 - A @ x_ls, 4))

# %% cell 3 — 수치적 rank
rng = np.random.default_rng(0)
c1, c2 = rng.normal(size=100), rng.normal(size=100)        # (100,), (100,)
exact = np.column_stack([c1, c2, c1 + c2])                 # (100, 3)
noisy = np.column_stack([c1, c2, c1 + c2 + 1e-9 * rng.normal(size=100)])

for name, X in [("정확히 종속", exact), ("1e-9 잡음", noisy)]:
    s = np.linalg.svd(X, compute_uv=False)                 # 특이값(7.6장)
    print(f"{name}: 특이값 {s[0]:.2f} {s[1]:.2f} {s[2]:.1e}  "
          f"rank(기본 tol) = {np.linalg.matrix_rank(X)}  "
          f"rank(tol=1e-6) = {np.linalg.matrix_rank(X, tol=1e-6)}")

# %% cell 4 — 저랭크 업데이트의 파라미터 수
d, k, r = 1024, 1024, 8
rng = np.random.default_rng(1)
B = rng.normal(size=(d, r))                  # (d, r)
A_lr = rng.normal(size=(r, k))               # (r, k)
dW = B @ A_lr                                # (d, k) 저랭크 업데이트
print("rank(dW)            =", np.linalg.matrix_rank(dW))
print("전체 행렬 파라미터 수 =", d * k)
print("두 인수 파라미터 수   =", d * r + r * k, f"({(d * r + r * k) / (d * k):.2%})")

# %% cell 5 — rank가 부족한 회귀와 ridge
rng = np.random.default_rng(2)
temp_c = rng.uniform(10, 40, size=50)                      # (50,) 섭씨
temp_f = 1.8 * temp_c + 32.0                               # (50,) 같은 온도의 화씨
X = np.column_stack([np.ones(50), temp_c, temp_f])         # (50, 3), 열이 종속
y_obs = 0.5 * temp_c + rng.normal(scale=0.3, size=50)      # (50,)
print("rank(X) =", np.linalg.matrix_rank(X))

n_vec = np.array([32.0, 1.8, -1.0])                        # X @ n_vec = 0
w_a = np.linalg.lstsq(X, y_obs)[0]
w_b = w_a + 10.0 * n_vec
print("w_a =", np.round(w_a, 3), " w_b =", np.round(w_b, 3))
print("예측 차이 최대 =", f"{np.abs(X @ w_a - X @ w_b).max():.1e}")

for lam in [1e-6, 1e-2, 1e2]:
    w = np.linalg.solve(X.T @ X + lam * np.eye(3), X.T @ y_obs)
    rmse = np.sqrt(np.mean((y_obs - X @ w) ** 2))
    print(f"lambda = {lam:6.0e}  w = {np.round(w, 3)}  |w| = {np.linalg.norm(w):.3f}  RMSE = {rmse:.4f}")
