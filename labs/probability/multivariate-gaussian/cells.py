"""11.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/probability/multivariate-gaussian/cells.py
"""
# %% cell 1 — Cholesky 샘플링과 Mahalanobis 거리
import numpy as np
from scipy import stats

mu = np.array([1.0, -1.0])                     # (2,)
Sigma = np.array([[1.0, 1.2],
                  [1.2, 2.25]])                # (2, 2) sigma1 = 1, sigma2 = 1.5, rho = 0.8
L = np.linalg.cholesky(Sigma)                  # (2, 2) 하삼각, Sigma = L L^T
rng = np.random.default_rng(0)
Z = rng.standard_normal((200_000, 2))          # (N, 2) 표준정규
X = mu + Z @ L.T                               # (N, 2) x = mu + L z
print("표본 평균   :", np.round(X.mean(axis=0), 3))
print("표본 공분산 :", np.round(np.cov(X, rowvar=False), 3).tolist())

D = X - mu                                     # (N, 2)
W = np.linalg.solve(L, D.T).T                  # (N, 2) 백색화 z = L^{-1}(x - mu)
d2 = (W**2).sum(axis=1)                        # (N,) Mahalanobis 거리 제곱
for r in (1.0, 2.0, 3.0):
    print(f"Mahalanobis <= {r}: 표본 {np.mean(d2 <= r**2):.4f}   chi2(2) {stats.chi2.cdf(r**2, df=2):.4f}"
          f"   | 유클리드 <= {r}: {np.mean((D**2).sum(1) <= r**2):.4f}")

# %% cell 2 — 조건부분포: 블록 공식과 샘플링
x1 = 2.2                                       # 관측한 x1 값
cond_mean = mu[1] + Sigma[1, 0] / Sigma[0, 0] * (x1 - mu[0])
cond_var = Sigma[1, 1] - Sigma[1, 0] ** 2 / Sigma[0, 0]     # Schur complement
band = np.abs(X[:, 0] - x1) < 0.05                          # x1 근처 띠 안의 표본
print(f"공식     E[x2 | x1={x1}] = {cond_mean:.3f}   Var = {cond_var:.3f}")
print(f"샘플링   E[x2 | x1≈{x1}] = {X[band, 1].mean():.3f}   Var = {X[band, 1].var():.3f}   (표본 {band.sum()}개)")
print(f"주변분포 E[x2] = {mu[1]:.3f}   Var = {Sigma[1, 1]:.3f}")

# %% cell 3 — 선형-가우시안 모델: 2초 뒤 위치 하나로 초기 위치·속도 추정
m0 = np.array([0.0, 1.0])                      # (2,) 사전 평균: 위치 m, 속도 m/s
P0 = np.diag([4.0, 1.0])                       # (2, 2) 사전 공분산
dt_s = 2.0
A = np.array([[1.0, dt_s]])                    # (1, 2) 2초 뒤 위치 = 위치 + dt * 속도
R = np.array([[0.25]])                         # (1, 1) 측정 잡음 분산
y = np.array([5.0])                            # (1,) 측정한 위치 m

S = A @ P0 @ A.T + R                           # (1, 1) y의 주변 공분산
K = P0 @ A.T @ np.linalg.inv(S)                # (2, 1) gain
m_post = m0 + K @ (y - A @ m0)
P_post = P0 - K @ A @ P0
print("사후 평균      :", np.round(m_post, 3))
print("사후 표준편차  :", np.round(np.sqrt(np.diag(P_post)), 3), "  (사전:", np.sqrt(np.diag(P0)), ")")
print("사후 상관계수  :", round(P_post[0, 1] / np.sqrt(P_post[0, 0] * P_post[1, 1]), 3))

J = np.block([[P0, P0 @ A.T], [A @ P0, S]])   # (3, 3) (x, y)의 결합 공분산
m_joint = m0 + J[:2, 2:] @ np.linalg.solve(J[2:, 2:], y - A @ m0)
print("결합분포를 조건화한 결과와 같은가:", np.allclose(m_joint, m_post))

# %% cell 4 — 정밀도 행렬의 0과 조건부 독립
phi, n = 0.8, 5                                # AR(1) 사슬 x1 -> x2 -> ... -> x5
C = phi ** np.abs(np.subtract.outer(np.arange(n), np.arange(n))) / (1 - phi**2)   # (5, 5) 공분산
Lam = np.linalg.inv(C)                         # (5, 5) 정밀도 행렬
print("공분산 (소수 둘째 자리):\n", np.round(C, 2))
print("정밀도 행렬:\n", np.round(Lam, 2) + 0.0)          # + 0.0: -0.0 표시를 0.0으로
d = np.sqrt(np.diag(Lam))
print("x1-x3 상관계수   :", round(C[0, 2] / np.sqrt(C[0, 0] * C[2, 2]), 3))
print("x1-x3 편상관계수 :", round(-Lam[0, 2] / (d[0] * d[2]), 3) + 0.0, " (나머지가 주어졌을 때)")

# %% cell 5 — PSD가 깨지는 공분산과 jitter
from scipy import linalg as sla

def try_cholesky(M, name):                     # scipy는 실패 시 항상 LinAlgError를 낸다
    try:
        sla.cholesky(M, lower=True)
        print(f"{name:24s} Cholesky 성공")
    except np.linalg.LinAlgError:
        print(f"{name:24s} Cholesky 실패")

t = np.linspace(0, 1, 50)                      # (50,) 입력 위치
K = np.exp(-0.5 * (t[:, None] - t[None, :]) ** 2 / 0.3**2)   # (50, 50) RBF 커널 행렬
w = np.linalg.eigvalsh(K)
print(f"RBF 커널: 최대 고윳값 {w.max():.1f},  |최소 고윳값| < 1e-12: {abs(w.min()) < 1e-12}")
try_cholesky(K + 1e-6 * np.eye(50), "RBF + jitter 1e-6")

C = np.array([[1.0, 0.9, 0.9],                 # 변수 쌍마다 다른 표본으로 추정한 상관계수
              [0.9, 1.0, 0.2],
              [0.9, 0.2, 1.0]])                # (3, 3)
w, Q = np.linalg.eigh(C)
print("쌍별 상관 행렬 고윳값:", np.round(w, 3))
try_cholesky(C + 1e-6 * np.eye(3), "쌍별 상관 + jitter 1e-6")
C_psd = Q @ np.diag(np.clip(w, 1e-3, None)) @ Q.T   # 음수 고윳값을 잘라 낸 PSD 근사
try_cholesky(C_psd, "고윳값 clip")
print("clip 뒤 최대 성분 변화:", round(np.abs(C_psd - C).max(), 3))
