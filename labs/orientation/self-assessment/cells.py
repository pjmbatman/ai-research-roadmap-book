"""0.4 본문 코드 셀(진단 문항). 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/orientation/self-assessment/cells.py
"""
# %% cell 1 — 선형대수: 곱의 rank 예측
import numpy as np

rng = np.random.default_rng(0)
U = rng.normal(size=(5, 2))     # (5, 2)
V = rng.normal(size=(2, 4))     # (2, 4)
A = U @ V                       # (5, 4)
print("shape:", A.shape, " rank:", np.linalg.matrix_rank(A))

# %% cell 2 — 미분: 해석적 gradient를 방향 미분으로 검증
N, D = 50, 3
X = rng.normal(size=(N, D))                  # (N, D)
y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=N)   # (N,)

def loss(w):                                 # w: (D,)
    r = X @ w - y                            # (N,)
    return r @ r / N

def grad(w):
    return 2.0 / N * X.T @ (X @ w - y)       # (D,)

w = rng.normal(size=D)                       # (D,)
v = rng.normal(size=D)                       # (D,) 임의의 방향
h = 1e-5
numeric = (loss(w + h * v) - loss(w - h * v)) / (2 * h)
print(f"grad·v = {grad(w) @ v:.8f}")
print(f"수치   = {numeric:.8f}")

# %% cell 3 — 확률: 표본 평균의 표준오차
sigma, n, trials = 2.0, 25, 20_000
samples = rng.normal(0.0, sigma, size=(trials, n))   # (trials, n)
means = samples.mean(axis=1)                         # (trials,)
print(f"표본 평균들의 표준편차 = {means.std():.4f}")
print(f"sigma / sqrt(n)       = {sigma / np.sqrt(n):.4f}")
