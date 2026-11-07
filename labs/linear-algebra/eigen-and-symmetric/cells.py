"""7.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/linear-algebra/eigen-and-symmetric/cells.py
"""
# %% cell 1 — 고유분해와 반복 적용
import numpy as np

A = np.array([[0.8, 0.3],
              [0.2, 0.7]])                     # (2, 2) 두 지역 사이 이동 비율(열의 합 = 1)
lam, V = np.linalg.eig(A)                      # 고윳값 (2,), 고유벡터 열 (2, 2)
lam, V = lam.real, V.real                      # 이 행렬의 고윳값은 실수다
print("고윳값        :", lam)
print("A v = lam v ? :", np.allclose(A @ V, V * lam))
print("A = V L V^-1 ?:", np.allclose(A, V @ np.diag(lam) @ np.linalg.inv(V)))

x0 = np.array([1.0, 0.0])                      # 처음에 모두 지역 1에
for k in [1, 5, 20]:
    xk = V @ np.diag(lam**k) @ np.linalg.solve(V, x0)
    print(f"k = {k:2d}  x_k = {np.round(xk, 4)}")
print("lam = 1 고유벡터(합 1로 정규화):", np.round(V[:, 0] / V[:, 0].sum(), 4))

# %% cell 2 — 대칭행렬과 비대칭 행렬의 고유벡터
rng = np.random.default_rng(0)
B = rng.normal(size=(4, 4))
S = (B + B.T) / 2                              # (4, 4) 대칭행렬
w, Q = np.linalg.eigh(S)                       # 실수 고윳값(오름차순), 정규직교 고유벡터
print("대칭  고윳값:", np.round(w, 3))
print("Q^T Q = I ?  ", np.allclose(Q.T @ Q, np.eye(4)), "  S = Q W Q^T ?", np.allclose(S, Q @ np.diag(w) @ Q.T))

_, V = np.linalg.eig(np.array([[2.0, 1.0], [0.0, 3.0]]))
V = V.real                                     # 고윳값 2, 3 (실수)
print("비대칭 고유벡터 사이 cos =", round(abs(V[:, 0] @ V[:, 1]), 4))
theta = np.pi / 6
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
print("30도 회전의 고윳값:", np.round(np.linalg.eigvals(R), 4))

# %% cell 3 — 양의 정부호성과 Rayleigh quotient
S2 = np.array([[2.0, 1.0], [1.0, 2.0]])
T2 = np.array([[1.0, 2.0], [2.0, 1.0]])
print("S2 고윳값:", np.linalg.eigvalsh(S2), "  T2 고윳값:", np.linalg.eigvalsh(T2))
x = np.array([1.0, -1.0])
print("x^T S2 x =", x @ S2 @ x, "  x^T T2 x =", x @ T2 @ x)

X = np.random.default_rng(1).normal(size=(10000, 2))           # (N, 2) 무작위 방향
rq = np.einsum("ni,ij,nj->n", X, S2, X) / np.einsum("ni,ni->n", X, X)
print(f"Rayleigh quotient 범위: [{rq.min():.4f}, {rq.max():.4f}]")
for M, name in [(S2, "S2"), (T2, "T2")]:
    try:
        np.linalg.cholesky(M)
        print(name, "Cholesky 성공 → 양의 정부호")
    except np.linalg.LinAlgError:
        print(name, "Cholesky 실패 → 양의 정부호 아님")

# %% cell 4 — 공분산 행렬의 주축
rng = np.random.default_rng(2)
temp_c = rng.normal(20.0, 3.0, size=500)                       # (500,) 기온
rh_pct = 60.0 - 2.0 * (temp_c - 20.0) + rng.normal(0.0, 4.0, size=500)   # (500,) 상대습도
D = np.column_stack([temp_c, rh_pct])                          # (500, 2)
C = np.cov(D, rowvar=False)                                    # (2, 2) 표본 공분산
w, Q = np.linalg.eigh(C)
print("공분산:\n", np.round(C, 2))
print("고윳값(분산):", np.round(w, 2))
print("제1주축     :", np.round(Q[:, 1], 3), " 기울기 =", round(Q[1, 1] / Q[0, 1], 3))
proj = (D - D.mean(0)) @ Q[:, 1]
print("제1주축 방향 표본 분산 =", round(proj.var(ddof=1), 2))

# %% cell 5 — 선형 동역학의 안정성
def run(A, x0, T):
    xs = [x0]
    for _ in range(T):
        xs.append(A @ xs[-1])
    return np.array([np.linalg.norm(x) for x in xs])           # (T + 1,)

x0 = np.array([1.0, 1.0])
stable = np.array([[0.9, 0.0], [0.0, 0.5]])
unstable = np.array([[1.05, 0.0], [0.0, 0.5]])
nonnormal = np.array([[0.9, 5.0], [0.0, 0.8]])                  # 고윳값 0.9, 0.8
for name, M in [("stable", stable), ("unstable", unstable), ("nonnormal", nonnormal)]:
    rho = max(abs(np.linalg.eigvals(M)))
    n = run(M, x0, 60)
    print(f"{name:9s} rho = {rho:.2f}  |x_10| = {n[10]:7.3f}  |x_60| = {n[60]:9.3e}  최대 = {n.max():.2f}")
