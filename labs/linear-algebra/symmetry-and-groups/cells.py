"""7.10 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/linear-algebra/symmetry-and-groups/cells.py
"""
# %% cell 1 — 군과 표현: 회전·치환·반사
import numpy as np
from itertools import permutations

def rot(theta):                                     # 2차원 회전의 표현 ρ(θ), (2, 2)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

a, b = 0.7, -1.9
print("ρ(a) ρ(b) = ρ(a + b) :", np.allclose(rot(a) @ rot(b), rot(a + b)))

C4 = [rot(k * np.pi / 2) for k in range(4)]         # 90° 회전 4개
in_C4 = lambda M: any(np.allclose(M, g) for g in C4)
print("C4 닫힘              :", all(in_C4(g @ h) for g in C4 for h in C4))

S3 = [np.eye(3)[list(p)] for p in permutations(range(3))]   # 치환 행렬 6개, 각 (3, 3)
in_S3 = lambda M: any(np.array_equal(M, P) for P in S3)
print("S3 원소 수, 닫힘     :", len(S3), all(in_S3(P @ Q) for P in S3 for Q in S3))
print("치환 행렬은 직교     :", all(np.allclose(P.T @ P, np.eye(3)) for P in S3))

F1 = np.diag([1.0, -1.0])                           # x축에 대한 반사
F2 = rot(0.6) @ F1 @ rot(-0.6)                      # 기울어진 축에 대한 반사
print("det F1, det(F1 F2)   :", round(np.linalg.det(F1)), round(np.linalg.det(F1 @ F2)))

# %% cell 2 — 불변·등변·둘 다 아님
rng = np.random.default_rng(0)
X = rng.normal(size=(5, 2))                         # 점구름 (N, D)
R, t = rot(1.1), np.array([0.5, -2.0])
gX = X @ R.T + t                                    # g·X: 모든 점을 회전한 뒤 평행이동

def pair_dists(X):                                  # 불변 후보: 정렬한 쌍별 거리 (N(N-1)/2,)
    d = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
    return np.sort(d[np.triu_indices(len(X), 1)])

centroid = lambda X: X.mean(axis=0)                 # 등변 후보: 무게중심 (D,)
W = rng.normal(size=(10, 2))
mlp = lambda X: np.tanh(X.reshape(-1) @ W)          # 좌표를 펼쳐 넣는 일반 MLP 층 (2,)

print("거리   불변  f(gX) = f(X)       :", np.allclose(pair_dists(gX), pair_dists(X)))
print("중심   등변  f(gX) = R f(X) + t :", np.allclose(centroid(gX), R @ centroid(X) + t))
print("MLP    불변                     :", np.allclose(mlp(gX), mlp(X)))
print("MLP    등변(회전만)             :", np.allclose(mlp(X @ R.T), R @ mlp(X)))

# %% cell 3 — 순열 등변 층과 불변 pooling
N, D_in, D_out = 6, 4, 3
H = rng.normal(size=(N, D_in))                      # 집합 원소 N개의 특징 (N, D_in)
W_self = rng.normal(size=(D_in, D_out))
W_mean = rng.normal(size=(D_in, D_out))
bias = rng.normal(size=D_out)

def equi_layer(H):                                  # (N, D_in) → (N, D_out)
    return H @ W_self + H.mean(axis=0, keepdims=True) @ W_mean + bias

P = np.eye(N)[rng.permutation(N)]                   # 치환 행렬 (N, N)
print("등변  f(PH) = P f(H)      :", np.allclose(equi_layer(P @ H), P @ equi_layer(H)))
print("불변  sum f(PH) = sum f(H):", np.allclose(equi_layer(P @ H).sum(0), equi_layer(H).sum(0)))

W_flat = rng.normal(size=(N * D_in, N * D_out))     # 펼친 집합에 쓰는 일반 dense 층
dense = lambda H: (H.reshape(-1) @ W_flat).reshape(N, D_out)
print("dense 층 등변             :", np.allclose(dense(P @ H), P @ dense(H)))
print("파라미터 수  등변 층 =", W_self.size + W_mean.size + bias.size, "  dense 층 =", W_flat.size)

# %% cell 4 — 회전 대칭과 각운동량 보존 (Noether)
def forces(x, k_ext):                               # x (N, 2) [m] → 힘 (N, 2) [N]
    diff = x[:, None, :] - x[None, :, :]            # (N, N, 2)
    f = -diff.sum(axis=1)                           # 쌍별 용수철(상수 1 N/m): 중심력, 회전 불변
    f[:, 0] -= k_ext * x[:, 0]                      # 외부장 V = k x²/2: x 방향만, 회전 불변 아님
    return f

def simulate(k_ext, steps=2000, dt_s=0.01):         # 질량 1 kg, velocity Verlet
    x = np.array([[1.0, 0.0], [-0.5, 0.8], [-0.4, -0.9]])
    v = np.array([[0.0, 0.6], [-0.5, -0.2], [0.4, -0.3]])
    L_z = lambda x, v: np.sum(x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0])   # 각운동량 [kg m²/s]
    L = [L_z(x, v)]
    for _ in range(steps):
        v_half = v + 0.5 * dt_s * forces(x, k_ext)
        x = x + dt_s * v_half
        v = v_half + 0.5 * dt_s * forces(x, k_ext)
        L.append(L_z(x, v))
    return np.array(L)

for k_ext in [0.0, 0.5]:
    L = simulate(k_ext)
    dL = np.abs(L - L[0]).max()
    print(f"외부장 k = {k_ext}:  L(0) = {L[0]:.4f}  최대 |L(t) - L(0)| =", "< 1e-12" if dL < 1e-12 else f"{dL:.2f}")
