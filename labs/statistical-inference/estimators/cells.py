"""12.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/statistical-inference/estimators/cells.py
"""
# %% cell 1 — 추정량은 확률변수다
import numpy as np

rng = np.random.default_rng(0)
mu_hz, sigma_hz, n = 2.0, 0.5, 25        # 모집단: 고유진동수 측정값의 분포

x = rng.normal(mu_hz, sigma_hz, size=n)   # 실제로 손에 쥔 표본 하나, (n,)
print(f"이 표본의 추정값  x̄ = {x.mean():.3f} Hz,  추정 표준오차 s/√n = {x.std(ddof=1) / np.sqrt(n):.3f} Hz")

reps = rng.normal(mu_hz, sigma_hz, size=(10000, n))   # 같은 실험을 10,000번 반복, (R, n)
xbar = reps.mean(axis=1)                               # 추정량의 표본분포, (R,)
print(f"반복 실험의 x̄    평균 {xbar.mean():.3f} Hz,  표준편차 {xbar.std():.3f} Hz")
print(f"이론값            평균 {mu_hz:.3f} Hz,  σ/√n     {sigma_hz / np.sqrt(n):.3f} Hz")

# %% cell 2 — 분산 추정량의 bias·variance·MSE
sigma2, n = 1.0, 5
X = rng.normal(0.0, np.sqrt(sigma2), size=(200000, n))   # (R, n)
ss = ((X - X.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)   # 편차 제곱합, (R,)

print("나누는 수   bias     variance   MSE      bias²+variance")
for d in (n - 1, n, n + 1):
    est = ss / d
    bias, var = est.mean() - sigma2, est.var()
    mse = np.mean((est - sigma2) ** 2)
    print(f"n{d - n:+d} = {d}    {bias:+.4f}  {var:.4f}     {mse:.4f}   {bias**2 + var:.4f}")

# %% cell 3 — 불편성과 일관성은 다른 성질이다
print("   n    X₁의 sd   x̄의 sd   σ̂²(÷n)의 bias  σ̂²(÷n)의 sd")
for n in (10, 100, 1000):
    X = rng.normal(0.0, 1.0, size=(5000, n))   # 참값: 평균 0, 분산 1, (R, n)
    v = X.var(axis=1)                          # 나누는 수 n (np.var 기본값)
    print(f"{n:5d}    {X[:, 0].std():.3f}     {X.mean(axis=1).std():.3f}     {v.mean() - 1:+.4f}        {v.std():.4f}")

# %% cell 4 — 효율성: 평균과 중앙값, 그리고 Cramér–Rao 하한
n, R = 100, 20000
samples = {
    "정규 N(0,1)":     rng.normal(0.0, 1.0, size=(R, n)),     # CRLB = 1/n
    "Laplace(0,1)":    rng.laplace(0.0, 1.0, size=(R, n)),    # CRLB = b²/n = 1/n
}
print("분포            n·Var(평균)  n·Var(중앙값)  n·CRLB")
for name, X in samples.items():
    print(f"{name:14s}  {n * X.mean(axis=1).var():.3f}        {n * np.median(X, axis=1).var():.3f}          1.000")
