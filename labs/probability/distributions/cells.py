"""11.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/probability/distributions/cells.py
"""
# %% cell 1 — 드문 사건의 수: binomial에서 Poisson으로
import numpy as np
from scipy import stats

lam = 3.0                                      # 단위 시간당 평균 사건 수
k = np.arange(0, 15)                           # (15,)
for n in (10, 100, 1000):
    gap = np.abs(stats.binom.pmf(k, n, lam / n) - stats.poisson.pmf(k, lam)).max()
    print(f"n = {n:5d}, p = {lam / n:.4f}:  max |Bin - Poisson| = {gap:.5f}")

# %% cell 2 — Poisson 과정: 사건 수와 대기 시간
rng = np.random.default_rng(0)
rate_per_s = 3.0
gaps_s = rng.exponential(1 / rate_per_s, size=(20_000, 40))   # (R, 40) 사건 간격
times_s = gaps_s.cumsum(axis=1)                               # (R, 40) 사건 시각
counts = (times_s <= 1.0).sum(axis=1)                         # (R,) 1초 동안의 사건 수
print(f"1초 사건 수: 평균 {counts.mean():.3f}  분산 {counts.var():.3f}   (Poisson(3): 둘 다 3)")
print(f"P(N = 0) ≈ {np.mean(counts == 0):.4f}   이론 {stats.poisson.pmf(0, 3):.4f}")

T = gaps_s.ravel()                                            # (R*40,) 대기 시간
for s in (0.5, 1.0):
    cond = np.mean(T[T > s] > s + 0.3)                        # P(T > s + 0.3 | T > s)
    print(f"P(T > {s} + 0.3 | T > {s}) ≈ {cond:.3f}   P(T > 0.3) = {np.exp(-rate_per_s * 0.3):.3f}")

# %% cell 3 — 지수족: log-partition의 미분이 평균과 분산
def check(name, A, T_mean, T_var, eta, h=1e-4):
    d1 = (A(eta + h) - A(eta - h)) / (2 * h)
    d2 = (A(eta + h) - 2 * A(eta) + A(eta - h)) / h**2
    print(f"{name:10s} A'(eta) = {d1:.4f}  E[T] = {T_mean:.4f}   A''(eta) = {d2:.4f}  Var[T] = {T_var:.4f}")

pi = 0.2                                       # Bernoulli: eta = logit(pi), A = log(1 + e^eta)
check("Bernoulli", lambda e: np.log1p(np.exp(e)), pi, pi * (1 - pi), np.log(pi / (1 - pi)))
lam = 4.0                                      # Poisson: eta = log(lam), A = e^eta
check("Poisson", np.exp, lam, lam, np.log(lam))

# %% cell 4 — 두꺼운 꼬리: 정규분포와 Student-t 적합 비교
x = stats.t.rvs(df=3, loc=10.0, scale=0.2, size=2000, random_state=np.random.default_rng(1))  # (N,) 측정값

mu, sd = stats.norm.fit(x)                     # 최대우도 적합
df, loc, sc = stats.t.fit(x, 5, loc=np.median(x), scale=x.std() / 2)   # 초기값: df 5
ll_n = stats.norm.logpdf(x, mu, sd).sum()
ll_t = stats.t.logpdf(x, df, loc, sc).sum()
print(f"정규  mu = {mu:.3f}  sd = {sd:.3f}             logL = {ll_n:8.1f}  AIC = {2 * 2 - 2 * ll_n:7.1f}")
print(f"t     loc = {loc:.3f} scale = {sc:.3f}  df = {df:.2f}  logL = {ll_t:8.1f}  AIC = {2 * 3 - 2 * ll_t:7.1f}")

thr = 1.0                                      # 중심에서 1.0 이상 벗어날 확률
print(f"P(|X - 10| > {thr}):  표본 {np.mean(np.abs(x - 10) > thr):.4f}   "
      f"정규 적합 {2 * stats.norm.sf(thr, 0, sd):.5f}   t 적합 {2 * stats.t.sf(thr / sc, df):.4f}")
