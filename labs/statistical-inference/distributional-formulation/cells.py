"""12.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/statistical-inference/distributional-formulation/cells.py
"""
# %% cell 1 — 손실이 고르는 대표값
import numpy as np

rng = np.random.default_rng(0)
y = rng.exponential(scale=1.0, size=2000)     # 오른쪽 꼬리가 긴 측정 잡음

grid = np.linspace(0, 4, 801)
def argmin(loss):
    return grid[np.argmin([loss(q) for q in grid])]

tau = 0.9
q_sq = argmin(lambda q: np.mean((y - q) ** 2))
q_ab = argmin(lambda q: np.mean(np.abs(y - q)))
q_pb = argmin(lambda q: np.mean((y - q) * (tau - (y < q))))

print(f"제곱손실 최소  {q_sq:.3f}   표본 평균       {y.mean():.3f}")
print(f"절댓값손실 최소 {q_ab:.3f}   표본 중앙값     {np.median(y):.3f}")
print(f"pinball(0.9)  {q_pb:.3f}   표본 90% 분위수 {np.quantile(y, tau):.3f}")

# %% cell 2 — 이분산 가우시안 회귀
rng = np.random.default_rng(1)
x = rng.uniform(0, 1, size=2000)
y = 2.0 * x + rng.normal(size=2000) * (0.1 + 0.5 * x)   # 잡음이 x에 따라 커진다

w, c = np.polyfit(x, y, 1)                  # 평균은 최소제곱으로 시작
alpha, beta = 0.0, np.log(np.std(y - (w * x + c)))
lr = 0.1
for _ in range(3000):
    mu = w * x + c
    s = alpha * x + beta                    # log sigma
    r2 = (y - mu) ** 2 * np.exp(-2 * s)
    d_mu = -(y - mu) * np.exp(-2 * s)       # dNLL/dmu
    d_s = 1.0 - r2                          # dNLL/d(log sigma)
    w, c = w - lr * np.mean(d_mu * x), c - lr * np.mean(d_mu)
    alpha, beta = alpha - lr * np.mean(d_s * x), beta - lr * np.mean(d_s)

for x0 in (0.1, 0.5, 0.9):
    print(f"x = {x0}:  평균 {w * x0 + c:.3f}   sigma {np.exp(alpha * x0 + beta):.3f}   (참값 sigma {0.1 + 0.5 * x0:.3f})")

# %% cell 3 — 적정 점수 규칙: NLL과 CRPS
from math import erf, pi, sqrt

Phi = np.vectorize(lambda z: 0.5 * (1 + erf(z / sqrt(2))))
def crps_gauss(y, mu, s):
    z = (y - mu) / s
    pdf = np.exp(-z**2 / 2) / sqrt(2 * pi)
    return np.mean(s * (z * (2 * Phi(z) - 1) + 2 * pdf - 1 / sqrt(pi)))

def nll_gauss(y, mu, s):
    return np.mean(0.5 * np.log(2 * pi * s**2) + (y - mu) ** 2 / (2 * s**2))

y = np.random.default_rng(2).normal(size=5000)     # 참 분포: N(0, 1)
for s in (0.3, 1.0, 3.0):
    print(f"예측 N(0, {s}^2)  NLL {nll_gauss(y, 0, s):.3f}   CRPS {crps_gauss(y, 0, s):.3f}")
