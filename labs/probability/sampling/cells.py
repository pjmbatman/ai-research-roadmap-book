"""11.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/probability/sampling/cells.py
"""
# %% cell 1 — Monte Carlo 오차는 1/√N으로 준다
import numpy as np

rng = np.random.default_rng(0)
exact = np.exp(-0.5)                       # E[cos X], X ~ N(0, 1)
print("      N     추정값    |오차|     표준오차 추정")
for N in (100, 10_000, 1_000_000):
    fx = np.cos(rng.normal(size=N))        # (N,)
    est, se = fx.mean(), fx.std(ddof=1) / np.sqrt(N)
    print(f"{N:>9,d}  {est:.5f}  {abs(est - exact):.5f}   {se:.5f}")

# %% cell 2 — 역변환 표집과 기각 표집
u = rng.uniform(size=100_000)
rate_per_s = 2.0
wait_s = -np.log(1 - u) / rate_per_s       # 지수분포 역 CDF: F^{-1}(u)
print(f"역변환 지수분포: 평균 {wait_s.mean():.4f} s (이론 {1 / rate_per_s}),  분산 {wait_s.var():.4f} (이론 {1 / rate_per_s**2})")

def beta25(x):                             # Beta(2, 5) 밀도, 최댓값 2.4576 (x = 0.2)
    return 30 * x * (1 - x) ** 4

M = 2.5                                    # 균등 제안분포의 envelope 상수
x = rng.uniform(size=100_000)              # 제안 표본
keep = rng.uniform(size=100_000) * M < beta25(x)
samples = x[keep]
print(f"기각 표집 Beta(2,5): 수용률 {keep.mean():.4f} (이론 1/M = {1 / M}),  평균 {samples.mean():.4f} (이론 {2 / 7:.4f})")

# %% cell 3 — 희귀 사건 확률: naive MC와 importance sampling
from scipy import stats

p_true = stats.norm.sf(4.0)                # P(X > 4), X ~ N(0, 1)
N = 100_000
x = rng.normal(size=N)
print(f"참값          {p_true:.3e}")
print(f"naive MC      {np.mean(x > 4):.3e}   (4 초과 표본 {np.sum(x > 4)}개)")

z = rng.normal(loc=4.0, size=N)            # 제안분포 q = N(4, 1)
w = stats.norm.pdf(z) / stats.norm.pdf(z, loc=4.0)          # p(z) / q(z)
h = w * (z > 4)                            # (N,) 가중치를 곱한 지시함수
est, se = h.mean(), h.std(ddof=1) / np.sqrt(N)
n_naive = p_true * (1 - p_true) / se**2    # 같은 표준오차에 필요한 naive 표본 수
print(f"IS, q=N(4,1)  {est:.3e} ± {se:.1e}   (naive로 같은 정밀도를 얻으려면 약 {n_naive:.1e}개)")

# %% cell 4 — 제안분포와 유효 표본 크기
N, R = 10_000, 200                         # E_p[X^2] = 1, p = N(0, 1)
print("제안분포 q      ESS/N(중앙값)   추정값 중앙값   5% ~ 95% (R번 반복)")
for name, loc, scale in [("N(0, 2^2)", 0, 2.0), ("N(2, 1)", 2, 1.0), ("N(0, 0.5^2)", 0, 0.5)]:
    ests, ess = [], []
    for r in range(R):
        z = rng.normal(loc, scale, size=N)
        logw = stats.norm.logpdf(z) - stats.norm.logpdf(z, loc, scale)
        w = np.exp(logw - logw.max())
        w /= w.sum()                       # 자기 정규화 가중치, 합 1
        ests.append(np.sum(w * z**2))
        ess.append(1 / np.sum(w**2))       # 유효 표본 크기
    lo, mid, hi = np.quantile(ests, [0.05, 0.5, 0.95])
    print(f"{name:13s}   {np.median(ess) / N:.3f}           {mid:.3f}          {lo:.3f} ~ {hi:.3f}")

# %% cell 5 — 기울기 추정: 재매개변수화 vs score function
mu, sigma, N, R = 1.0, 1.0, 100, 2_000     # d/dmu E[X^2] = 2 mu = 2
eps = rng.normal(size=(R, N))              # (R, N)
x = mu + sigma * eps                       # 재매개변수화: x = mu + sigma * eps
g_rep = (2 * x).mean(axis=1)               # d(x^2)/dmu = 2x · dx/dmu
g_sf = (x**2 * (x - mu) / sigma**2).mean(axis=1)            # f(x) · d log p / d mu
g_sfb = ((x**2 - (mu**2 + sigma**2)) * (x - mu) / sigma**2).mean(axis=1)   # baseline E[f] 빼기
for name, g in [("재매개변수화", g_rep), ("score function", g_sf), ("score + baseline", g_sfb)]:
    print(f"{name:18s} 평균 {g.mean():.4f}   표준편차 {g.std():.4f}")
