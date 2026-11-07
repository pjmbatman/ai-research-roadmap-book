"""12.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/statistical-inference/likelihood/cells.py
"""
# %% cell 1 — 같은 식, 다른 변수: 확률과 우도
import numpy as np
from math import comb

n, k = 20, 7                                              # 20번 시도 중 7번 성공
theta = np.linspace(0, 1, 100001)                         # 파라미터 격자, (G,)
lik = comb(n, k) * theta**k * (1 - theta) ** (n - k)      # L(θ) = p(y=7 | θ), (G,)

ys = np.arange(n + 1)                                     # 가능한 모든 결과, (21,)
prob = np.array([comb(n, y) for y in ys]) * 0.35**ys * 0.65 ** (n - ys)   # p(y | θ=0.35)
print(f"θ = 0.35 고정, y에 대한 합   Σ_y p(y | θ) = {prob.sum():.4f}")
print(f"y = 7 고정, θ에 대한 적분    ∫ L(θ) dθ   = {lik.mean():.4f}")   # [0, 1] 균등 격자의 평균
print(f"우도를 최대화하는 θ          θ̂ = {theta[np.argmax(lik)]:.3f}   (k/n = {k / n})")

# %% cell 2 — 왜 로그우도인가
rng = np.random.default_rng(0)
t_s = rng.exponential(scale=2.0, size=1000)    # 붕괴까지 걸린 시간, 참 λ = 0.5 /s, (N,)

def loglik(lam):                               # Σ_i log(λ e^{-λ t_i})
    return np.sum(np.log(lam) - lam * t_s)

print("우도 (밀도의 곱)    ", np.prod(0.5 * np.exp(-0.5 * t_s)))
print("로그우도 (로그의 합)", round(loglik(0.5), 2))
lams = np.linspace(0.3, 0.8, 5001)
lam_grid = lams[np.argmax([loglik(l) for l in lams])]
print(f"격자에서 최대화한 λ̂ = {lam_grid:.4f} /s,  닫힌 해 1/t̄ = {1 / t_s.mean():.4f} /s")

# %% cell 3 — cross-entropy는 categorical 음의 로그우도다
p = np.array([[0.7, 0.2, 0.1],
              [0.1, 0.8, 0.1],
              [0.3, 0.3, 0.4]])                     # 모델이 낸 클래스 확률, (N, K)
y = np.array([0, 1, 2])                             # 정답 label, (N,)

nll = -np.mean(np.log(p[np.arange(len(y)), y]))     # -1/N Σ log p(y_i | x_i)
onehot = np.eye(3)[y]                               # (N, K)
ce = -np.mean(np.sum(onehot * np.log(p), axis=1))   # -1/N Σ_i Σ_k y_ik log p_ik
print(f"categorical NLL {nll:.4f}   cross-entropy {ce:.4f}")

# %% cell 4 — MAP = 정규화: 가우시안 사전분포와 ridge
rng = np.random.default_rng(1)
f = lambda x: np.sin(2 * np.pi * x)
x = np.sort(rng.uniform(0, 1, 12))                 # 측정 12개, (N,)
y = f(x) + rng.normal(0, 0.2, size=12)
x_te = np.linspace(0, 1, 200)
Phi = lambda x: np.vander(x, 10, increasing=True)  # 9차 다항식 특징, (N, 10)

sigma, tau = 0.2, 10.0         # 잡음 표준편차, 사전분포 w ~ N(0, τ²I)의 표준편차
lam = sigma**2 / tau**2        # MAP의 정규화 강도 λ = σ²/τ²
A = Phi(x)
w_mle = np.linalg.lstsq(A, y, rcond=None)[0]
w_map = np.linalg.solve(A.T @ A + lam * np.eye(10), A.T @ y)
for name, w in (("MLE", w_mle), ("MAP", w_map)):
    tr = np.sqrt(np.mean((A @ w - y) ** 2))
    te = np.sqrt(np.mean((Phi(x_te) @ w - f(x_te)) ** 2))
    print(f"{name}  학습 RMSE {tr:.3f}   참 함수 대비 RMSE {te:.3f}   max|w| {np.abs(w).max():8.1f}")
