"""11.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/probability/random-variables/cells.py
"""
# %% cell 1 — 혼합형 확률변수: 하루 강수량
import numpy as np

rng = np.random.default_rng(0)
n_days = 100_000
wet = rng.random(n_days) < 0.4                                          # (N,) 비가 온 날
rain_mm = np.where(wet, rng.exponential(scale=5.0, size=n_days), 0.0)   # (N,) 강수량

print(f"P(R = 0)    ≈ {np.mean(rain_mm == 0):.3f}   (이론 0.600)")
print(f"P(R = 3 mm) ≈ {np.mean(rain_mm == 3.0):.3f}   (연속 부분의 한 점)")
for r in (0.0, 1.0, 5.0, 20.0):
    F_theory = 0.6 + 0.4 * (1 - np.exp(-r / 5.0))
    print(f"F({r:4.1f} mm) ≈ {np.mean(rain_mm <= r):.3f}   이론 {F_theory:.3f}")

# %% cell 2 — PDF 값은 확률이 아니다
from scipy import stats

X = stats.uniform(loc=0.0, scale=0.5)          # 0~0.5 균등분포
print("pdf(0.2) =", X.pdf(0.2))
print("P(0.1 < X <= 0.3) = F(0.3) - F(0.1) =", round(X.cdf(0.3) - X.cdf(0.1), 3))

x = X.rvs(size=200_000, random_state=np.random.default_rng(1))   # (N,)
print("표본에서 센 비율                    =", round(np.mean((x > 0.1) & (x <= 0.3)), 3))
for h in (0.1, 0.01, 0.001):
    frac = np.mean(np.abs(x - 0.2) < h / 2)                       # 폭 h 구간에 든 비율
    print(f"h = {h:<5}  P(|X - 0.2| < h/2) ≈ {frac:.4f}   비율 / h = {frac / h:.2f}")

# %% cell 3 — CDF와 quantile function
T = stats.expon(scale=2000.0)                  # 부품 수명(시간), 평균 2000 h
for p in (0.1, 0.5, 0.9):
    t_h = T.ppf(p)                             # quantile function F^{-1}(p)
    print(f"F^-1({p}) = {t_h:7.1f} h   F(F^-1(p)) = {T.cdf(t_h):.3f}")

K = stats.poisson(mu=3.0)                      # 이산 분포: CDF가 계단
print("Poisson(3)  F(2) =", round(K.cdf(2), 3), "  F(3) =", round(K.cdf(3), 3))
print("F^-1(0.5) =", K.ppf(0.5), "  (F(k) >= 0.5인 가장 작은 k)")

u = np.random.default_rng(2).random(100_000)   # (N,) Unif(0, 1)
t_sim = T.ppf(u)                               # 역변환 샘플링
print(f"F^-1(U)의 표본 중앙값 {np.median(t_sim):.1f} h   이론 중앙값 {T.median():.1f} h")

# %% cell 4 — 변수변환: Y = exp(X)
X = stats.norm(0.0, 0.5)
Y = stats.lognorm(s=0.5)                       # exp(X)의 분포 (scipy의 로그정규)
y_grid = np.array([0.5, 1.0, 1.5, 2.0])
p_formula = X.pdf(np.log(y_grid)) / y_grid     # p_X(g^{-1}(y)) |d g^{-1}/dy|
print("공식   p_Y(y):", np.round(p_formula, 4))
print("scipy  p_Y(y):", np.round(Y.pdf(y_grid), 4))

y = np.exp(X.rvs(size=200_000, random_state=np.random.default_rng(3)))   # (N,)
print(f"중앙값  exp(X의 중앙값) = {np.exp(X.median()):.3f}   Y 표본 {np.median(y):.3f}")
print(f"평균    exp(X의 평균)   = {np.exp(X.mean()):.3f}   Y 표본 {y.mean():.3f}")
print(f"최빈값  exp(X의 최빈값) = {np.exp(0.0):.3f}   Y 이론 {np.exp(-0.5**2):.3f}")
