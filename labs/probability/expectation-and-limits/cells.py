"""11.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/probability/expectation-and-limits/cells.py
"""
# %% cell 1 — 선형성과 상관 0인 종속 변수
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=1_000_000)                # X ~ N(0, 1)
y = x**2                                      # Y는 X로 완전히 결정된다

print(f"E[X + Y]       = {np.mean(x + y):.4f}")
print(f"E[X] + E[Y]    = {np.mean(x) + np.mean(y):.4f}")
print(f"Cov(X, Y)      = {np.cov(x, y)[0, 1]:.4f}")
print(f"Var(X + Y)     = {np.var(x + y):.4f}   Var X + Var Y = {np.var(x) + np.var(y):.4f}")
print(f"P(Y > 4)       = {np.mean(y > 4):.4f}   P(Y > 4 | |X| < 1) = {np.mean(y[np.abs(x) < 1] > 4):.4f}")

# %% cell 2 — 전체 분산의 법칙: 배치마다 다른 보정 오차
n_batch, n_per = 2000, 50
offset_v = rng.normal(0.0, 0.3, size=n_batch)                  # (n_batch,) 배치별 보정 오차
noise_v = rng.normal(0.0, 0.4, size=(n_batch, n_per))          # (n_batch, n_per) 측정 잡음
reading_v = 10.0 + offset_v[:, None] + noise_v                 # (n_batch, n_per)

within = reading_v.var(axis=1).mean()      # E[Var(Y | 배치)]
between = reading_v.mean(axis=1).var()     # Var(E[Y | 배치])
print(f"전체 분산         {reading_v.var():.4f}")
print(f"배치 안 + 배치 간  {within:.4f} + {between:.4f} = {within + between:.4f}")
print(f"이론값            0.4^2 + 0.3^2 = {0.4**2 + 0.3**2:.4f}")

# %% cell 3 — Markov·Chebyshev 경계와 실제 꼬리확률
# 평균 1, 분산 1인 지수분포: P(X >= a) = exp(-a)
print(" a    P(X >= a)  Markov 1/a   |   k    P(|X-1| >= k)  Chebyshev 1/k^2")
for a, k in [(2, 1.5), (4, 2.0), (8, 3.0)]:
    tail = np.exp(-a)
    two_sided = np.exp(-(1 + k))           # k >= 1이면 X <= 1 - k인 쪽은 확률 0
    print(f"{a:2d}    {tail:.4f}     {1 / a:.4f}      |  {k:.1f}   {two_sided:.4f}         {1 / k**2:.4f}")

# %% cell 4 — Jensen 부등식: 로그의 평균과 평균의 로그
w = rng.lognormal(mean=0.0, sigma=1.0, size=1_000_000)   # 양수 확률변수
print(f"log E[W] = {np.log(np.mean(w)):.4f}   (이론값 0.5)")
print(f"E[log W] = {np.mean(np.log(w)):.4f}   (이론값 0.0)")

# %% cell 5 — 표본 평균의 분포: 지수분포 vs Cauchy 분포
reps = 20_000
print("        지수분포(평균 1, 표준편차 1)                Cauchy")
for n in (5, 30, 300):
    m_exp = rng.exponential(1.0, size=(reps, n)).mean(axis=1)    # (reps,)
    m_cau = rng.standard_cauchy(size=(reps, n)).mean(axis=1)     # (reps,)
    z = (m_exp - 1.0) * np.sqrt(n)                               # 표준화한 표본 평균
    lo, hi = np.mean(z < -1.96), np.mean(z > 1.96)               # 정규분포면 각각 0.025
    iqr = np.subtract(*np.quantile(m_cau, [0.75, 0.25]))
    skew = np.mean((m_exp - m_exp.mean()) ** 3) / m_exp.std() ** 3
    print(f"n = {n:3d}  왜도 {skew:.3f}  P(Z<-1.96) {lo:.4f}  P(Z>1.96) {hi:.4f}"
          f"   평균의 IQR {iqr:.3f}")
