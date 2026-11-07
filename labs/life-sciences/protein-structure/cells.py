"""29.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/life-sciences/protein-structure/cells.py
"""
# %% cell 1 — Cα 좌표, 거리 행렬, 접촉 지도
import numpy as np

def helix(n, start, sign):                             # 이상적인 α-나선의 Cα 좌표 (Å)
    t = np.arange(n)
    ang = np.deg2rad(100.0) * t                        # 잔기당 100° 회전
    return np.stack([2.3 * np.cos(ang), 2.3 * np.sin(ang), 1.5 * t * sign], 1) + start

h1 = helix(20, [0.0, 0.0, 0.0], +1)                    # (20, 3)
h2 = helix(20, [10.0, 0.0, 28.5], -1)                  # (20, 3) 반대 방향의 두 번째 나선
loop = h1[-1] + np.linspace(0, 1, 5)[1:-1, None] * (h2[0] - h1[-1])   # (3, 3)
X = np.vstack([h1, loop, h2])                          # (43, 3) Å

def dist_matrix(X):
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)   # (L, L)

D = dist_matrix(X)
sep = np.abs(np.subtract.outer(np.arange(len(X)), np.arange(len(X))))   # (L, L) 서열 간격 |i-j|
contact = (D < 8.0) & (sep >= 6)                       # 서열상 멀리 떨어진 접촉
print("나선 안 이웃 Cα 거리:", np.round(np.linalg.norm(np.diff(h1, axis=0), axis=1)[:3], 2), "Å")
print("장거리 접촉 쌍 수:", contact.sum() // 2)

rng = np.random.default_rng(0)
Rq, _ = np.linalg.qr(rng.normal(size=(3, 3)))          # 무작위 직교 행렬
X_moved = X @ Rq.T + np.array([5.0, -3.0, 12.0])       # 회전 + 평행이동
print("이동 후 좌표 최대 변화:", round(float(np.abs(X_moved - X).max()), 1), "Å")
print("이동 후 거리 행렬 변화 < 1e-8:", np.abs(dist_matrix(X_moved) - D).max() < 1e-8)

# %% cell 2 — Kabsch 정렬과 RMSD
def kabsch(P, Q, allow_reflection=False):              # P @ R + t가 Q에 가장 가깝도록
    P0, Q0 = P - P.mean(0), Q - Q.mean(0)              # (L, 3) 중심을 원점으로
    U, S, Vt = np.linalg.svd(P0.T @ Q0)                # 3x3 공분산의 SVD
    d = 1.0 if allow_reflection else np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt                # (3, 3)
    rmsd = np.sqrt(np.mean(np.sum((P0 @ R - Q0) ** 2, axis=1)))
    return rmsd, R

def plain_rmsd(P, Q):
    return np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1)))

model = X_moved + rng.normal(0, 0.5, X.shape)          # 좌표마다 0.5 Å 잡음
rmsd, R = kabsch(model, X)
print(f"정렬 없는 RMSD   {plain_rmsd(model, X):6.2f} Å")
print(f"Kabsch 정렬 RMSD {rmsd:6.2f} Å,  det(R) = {np.linalg.det(R):.3f}")

mirror = X * np.array([1.0, 1.0, -1.0])                # 거울상: 거리 행렬은 같다
print("거울상 거리 행렬 변화 < 1e-8:", np.abs(dist_matrix(mirror) - D).max() < 1e-8)
print(f"거울상, 반사 허용  RMSD {kabsch(mirror, X, allow_reflection=True)[0]:5.2f} Å")
print(f"거울상, 회전만     RMSD {kabsch(mirror, X)[0]:5.2f} Å")

# %% cell 3 — 경첩 운동: RMSD, TM-score, lDDT
def tm_score(P, Q):                                    # 여러 부분 구간에서 시작한 중첩 중 최댓값(하한)
    L = len(Q)
    d0 = 1.24 * (L - 15) ** (1 / 3) - 1.8
    best = 0.0
    for s in range(0, L - 7):                          # 길이 8 구간마다 Kabsch 중첩
        _, R = kabsch(P[s:s + 8], Q[s:s + 8])
        Pa = (P - P[s:s + 8].mean(0)) @ R + Q[s:s + 8].mean(0)
        best = max(best, np.mean(1 / (1 + (np.linalg.norm(Pa - Q, axis=1) / d0) ** 2)))
    return best

def lddt_ca(P, Q, r0=15.0):                            # Cα만 쓴 단순화 lDDT (중첩 없음)
    Dp, Dq = dist_matrix(P), dist_matrix(Q)
    pair = (Dq < r0) & ~np.eye(len(Q), dtype=bool)     # 기준 구조에서 15 Å 안의 쌍
    diff = np.abs(Dp - Dq)[pair]
    return np.mean([np.mean(diff < t) for t in (0.5, 1.0, 2.0, 4.0)])

theta = np.deg2rad(35.0)                               # 고리 시작점을 축으로 두 번째 나선을 35° 회전
Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
hinge = X.copy()
hinge[20:] = (X[20:] - X[20]) @ Ry.T + X[20]
noisy = X + rng.normal(0, 1.3, X.shape)                # 전체에 고르게 퍼진 오차
for name, P in [("경첩 운동", hinge), ("전체 잡음", noisy)]:
    print(f"{name}: RMSD {kabsch(P, X)[0]:4.2f} Å  TM-score {tm_score(P, X):.2f}  lDDT(Cα) {lddt_ca(P, X):.2f}")

# %% cell 4 — 잔기별 신뢰도와 실제 오차
from scipy.stats import spearmanr

sigma = np.full(len(X), 0.3)                           # (43,) 잔기별 실제 오차 크기 (Å)
sigma[18:26] = 2.5                                     # 고리 주변은 불확실
sigma[-5:] = 4.0                                       # 끝부분은 거의 모름
pred = X + rng.normal(size=X.shape) * sigma[:, None]
conf = np.clip(100 - 22 * sigma + rng.normal(0, 5, len(X)), 0, 100)   # 모델이 낸 신뢰도 점수

def lddt_per_res(P, Q, r0=15.0):                       # 잔기별 단순화 lDDT
    Dp, Dq = dist_matrix(P), dist_matrix(Q)
    pair = (Dq < r0) & ~np.eye(len(Q), dtype=bool)
    ok = np.mean([(np.abs(Dp - Dq) < t) for t in (0.5, 1.0, 2.0, 4.0)], axis=0)   # (L, L)
    return np.array([ok[i, pair[i]].mean() for i in range(len(Q))])

rho = spearmanr(conf, lddt_per_res(pred, X)).statistic
hi = conf >= 70
print(f"신뢰도와 잔기별 lDDT의 Spearman 상관: {rho:.2f}")
print(f"전체 {len(X)}개 잔기 RMSD {kabsch(pred, X)[0]:.2f} Å")
print(f"신뢰도 70 이상 {hi.sum()}개 잔기 RMSD {kabsch(pred[hi], X[hi])[0]:.2f} Å")
