"""7.8 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/linear-algebra/manifolds-and-geometry/cells.py
"""
# %% cell 1 — 구면의 chart와 접공간
import numpy as np

def sphere(u):                       # 국소 좌표 u = (θ, φ) [rad] → R^3 위의 점
    th, ph = u
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])

def jacobian(f, u, h=1e-6):          # 중심 차분 Jacobian, (출력 차원, 입력 차원)
    u = np.asarray(u, dtype=float)
    cols = [(f(u + h * e) - f(u - h * e)) / (2 * h) for e in np.eye(len(u))]
    return np.stack(cols, axis=1)

u0 = np.array([1.0, 0.5])
p = sphere(u0)                       # (3,)
J = jacobian(sphere, u0)             # (3, 2)
print("|p| = 1     :", np.isclose(np.linalg.norm(p), 1.0))
print("J shape     :", J.shape, " rank =", np.linalg.matrix_rank(J))
print("J^T p = 0   :", np.allclose(J.T @ p, 0, atol=1e-8))
print("북극의 rank :", np.linalg.matrix_rank(jacobian(sphere, [0.0, 0.5]), tol=1e-6))

# %% cell 2 — 회전군 SO(3)의 지수사상 (Rodrigues 공식)
from scipy.linalg import expm

def hat(w):                          # (3,) → (3, 3) 반대칭 행렬
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])

def rodrigues(w):                    # w = 회전축 방향 × 회전각 [rad]
    th = np.linalg.norm(w)
    K = hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

w = np.array([0.3, -0.5, 0.8])
R = rodrigues(w)                     # (3, 3)
print("expm(hat w)와 일치 :", np.allclose(R, expm(hat(w))))
print("R^T R = I          :", np.allclose(R.T @ R, np.eye(3)))
print("det R              :", np.linalg.det(R).round(6))
print("회전축 고정 R w = w:", np.allclose(R @ w, w))

h = 1e-6
dR = (rodrigues(h * w) - rodrigues(-h * w)) / (2 * h)   # 곡선 R(t)의 t = 0 속도
print("dR/dt = hat(w)     :", np.allclose(dR, hat(w), atol=1e-6))

# %% cell 3 — Swiss roll: 전역 PCA와 국소 PCA
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors

X, t_roll = make_swiss_roll(n_samples=1500, random_state=0)    # X (N, 3)
Xc = X - X.mean(axis=0)
s = np.linalg.svd(Xc, compute_uv=False)
print("전역 PCA 분산 비율:", np.round(s**2 / np.sum(s**2), 3))

k = 12
_, idx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)  # (N, k)
local = []
for nb in idx[::10]:                                           # 150개 점에서
    P = X[nb] - X[nb].mean(axis=0)                             # (k, 3)
    v = np.linalg.svd(P, compute_uv=False) ** 2
    local.append(v / v.sum())
print("국소 PCA 분산 비율 평균:", np.round(np.mean(local, axis=0), 3))

# %% cell 4 — 직선 보간과 측지선 보간
p = np.array([1.0, 0.0, 0.0])
q = np.array([0.0, 1.0, 0.0])
ang = np.arccos(p @ q)               # 두 단위벡터 사이 각 [rad]

for t in [0.25, 0.5, 0.75]:
    lerp = (1 - t) * p + t * q
    slerp = (np.sin((1 - t) * ang) * p + np.sin(t * ang) * q) / np.sin(ang)
    print(f"t = {t:.2f}  |lerp| = {np.linalg.norm(lerp):.3f}  |slerp| = {np.linalg.norm(slerp):.3f}")
print(f"측지 거리 = {ang:.4f}  직선 거리 = {np.linalg.norm(p - q):.4f}")

# %% cell 5 — rank와 다양체 차원이 다른 경우
t = np.linspace(-1, 1, 200)
curve = np.stack([t, t**2, t**3], axis=1)                      # (200, 3)
C = curve - curve.mean(axis=0)
print("특이값          :", np.round(np.linalg.svd(C, compute_uv=False), 2))
print("데이터 행렬 rank:", np.linalg.matrix_rank(C))

J_curve = np.stack([np.ones_like(t), 2 * t, 3 * t**2], axis=1)  # 각 점의 Jacobian (200, 3) = 200개의 (3, 1)
print("곡선 Jacobian rank (모든 점):", sorted({int(np.linalg.matrix_rank(j[:, None])) for j in J_curve}))
