"""7.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/linear-algebra/svd-and-pca/cells.py
"""
# %% cell 1 — SVD의 기하
import numpy as np

A = np.array([[3.0, 0.0],
              [4.0, 5.0]])                               # (2, 2)
U, s, Vt = np.linalg.svd(A)                              # (2, 2), (2,), (2, 2)
print("특이값          :", np.round(s, 4))
print("U S V^T = A ?   :", np.allclose(U @ np.diag(s) @ Vt, A))
print("A v1 = s1 u1 ?  :", np.allclose(A @ Vt[0], s[0] * U[:, 0]))
print("sqrt(eig(A^T A)):", np.round(np.sqrt(np.linalg.eigvalsh(A.T @ A))[::-1], 4))

theta = np.linspace(0, 2 * np.pi, 3601)
circle = np.stack([np.cos(theta), np.sin(theta)])        # (2, 3601) 단위원
r = np.linalg.norm(A @ circle, axis=0)                   # 상의 원점 거리
print(f"단위원의 상: 최대 반지름 {r.max():.4f}, 최소 반지름 {r.min():.4f}")
A_pinv = Vt.T @ np.diag(1 / s) @ U.T                     # V S^+ U^T
print("V S^+ U^T = pinv(A) ?", np.allclose(A_pinv, np.linalg.pinv(A)))

# %% cell 2 — 저랭크 근사와 Eckart–Young
rng = np.random.default_rng(0)
x, t = np.linspace(0, 1, 60), np.linspace(0, 1, 40)
X = (np.outer(np.sin(np.pi * x), np.exp(-t))
     + 0.5 * np.outer(np.sin(2 * np.pi * x), np.cos(3 * t))
     + 0.2 * np.outer(np.sin(3 * np.pi * x), t)
     + 0.01 * rng.normal(size=(60, 40)))                 # (60, 40) 세 패턴 + 잡음
U, s, Vt = np.linalg.svd(X, full_matrices=False)
print("특이값 앞 6개:", np.round(s[:6], 3))
for k in [1, 2, 3, 4]:
    Xk = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]              # (60, 40) rank-k 근사
    err = np.linalg.norm(X - Xk)
    G = np.linalg.qr(rng.normal(size=(60, k)))[0]        # 무작위 k차원 부분공간
    err_rand = np.linalg.norm(X - G @ (G.T @ X))
    energy = (s[:k] ** 2).sum() / (s ** 2).sum()
    print(f"k = {k}  오차 {err:.4f}  sqrt(꼬리 s^2 합) {np.sqrt((s[k:] ** 2).sum()):.4f}  "
          f"무작위 부분공간 {err_rand:.4f}  에너지 {energy:.4%}")

# %% cell 3 — SVD로 PCA, 공분산 고유분해와 비교
rng = np.random.default_rng(1)
N = 500
z = rng.normal(size=(N, 2))                               # (N, 2) 잠재 요인
W = np.array([[2.0, 0.0, 1.0], [0.0, 1.0, 1.0]])          # (2, 3)
D = z @ W + 0.1 * rng.normal(size=(N, 3)) + [1.0, 10.0, -2.0]   # (N, 3)

Dc = D - D.mean(axis=0)                                   # 중심화
U, s, Vt = np.linalg.svd(Dc, full_matrices=False)         # (N, 3), (3,), (3, 3)
var_svd = s ** 2 / (N - 1)
w, Q = np.linalg.eigh(np.cov(D, rowvar=False))            # 오름차순
print("SVD 분산 :", np.round(var_svd, 4))
print("eigh 분산:", np.round(w[::-1], 4))
print("주성분 방향 |cos|:", np.round(np.abs(np.sum(Vt * Q[:, ::-1].T, axis=1)), 6))
print("설명 분산 비율:", np.round(var_svd / var_svd.sum(), 4))
Z = Dc @ Vt.T / np.sqrt(var_svd)                          # (N, 3) whitening
print("whitening 후 공분산 = I ?", np.allclose(np.cov(Z, rowvar=False), np.eye(3)))

# %% cell 4 — 중심화를 빼먹으면
_, s_raw, Vt_raw = np.linalg.svd(D, full_matrices=False)  # 평균을 빼지 않음
mean_dir = D.mean(axis=0) / np.linalg.norm(D.mean(axis=0))   # (3,)
print("|cos(중심화 없는 제1방향, 평균 방향)| =", round(abs(Vt_raw[0] @ mean_dir), 4))
print("|cos(중심화한 제1방향,   평균 방향)| =", round(abs(Vt[0] @ mean_dir), 4))
print("중심화 없이 '분산' 비율:", np.round(s_raw ** 2 / (s_raw ** 2).sum(), 4))

# %% cell 5 — randomized SVD
rng = np.random.default_rng(2)
m, n, k, p = 1000, 200, 5, 10
Ua = np.linalg.qr(rng.normal(size=(m, n)))[0]             # (m, n)
Va = np.linalg.qr(rng.normal(size=(n, n)))[0]             # (n, n)
sv = 0.7 ** np.arange(n)                                  # 빠르게 감소하는 특이값
M = (Ua * sv) @ Va.T                                      # (m, n)

Omega = rng.normal(size=(n, k + p))                       # (n, k + p) 무작위 시험 벡터
Y = M @ (M.T @ (M @ Omega))                               # (m, k + p) 거듭제곱 1회
Qr = np.linalg.qr(Y)[0]                                   # (m, k + p) 상의 basis
s_small = np.linalg.svd(Qr.T @ M, compute_uv=False)       # (k + p, n) 작은 행렬의 SVD
print("참 특이값       :", np.round(sv[:k], 6))
print("randomized SVD  :", np.round(s_small[:k], 6))
print(f"최대 상대 오차  : {np.max(np.abs(s_small[:k] - sv[:k]) / sv[:k]):.1e}")
