"""21.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/reliability/evaluating-predictive-distributions/cells.py
"""
# %% cell 1 — 적정한 점수와 적정하지 않은 점수
import numpy as np

p_true = 0.7                                   # 사건이 실제로 일어날 확률
print("보고한 q   기대 Brier   기대 |q - y|")
for q in (0.5, 0.7, 0.9, 1.0):
    brier = p_true * (1 - q) ** 2 + (1 - p_true) * q ** 2
    absol = p_true * (1 - q) + (1 - p_true) * q
    print(f"  {q:.1f}      {brier:.3f}        {absol:.3f}")

# %% cell 2 — 표본(앙상블)으로 계산하는 CRPS
from scipy.stats import norm

def crps_samples(x, y):
    """x: (S,) 예측분포의 표본, y: 관측값. CRPS = E|X - y| - E|X - X'| / 2"""
    return np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))

def crps_gauss(y, mu, s):
    z = (y - mu) / s
    return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))

rng = np.random.default_rng(0)
x = rng.normal(0.0, 1.0, size=2000)            # (S,) N(0, 1) 예측분포에서 뽑은 표본
for y in (0.0, 1.0, 3.0):
    print(f"y = {y}  표본 CRPS {crps_samples(x, y):.3f}   닫힌 형태 {crps_gauss(y, 0, 1):.3f}")
print(f"표본 1개(점 예측 0)일 때 CRPS = |0 - 1| = {crps_samples(np.zeros(1), 1.0):.3f}")

# %% cell 3 — calibration·sharpness·조건별 coverage를 한 표로
xr = rng.uniform(0, 1, size=20000)             # (N,)
sd = 0.1 + 0.5 * xr                            # 참 잡음 sd: x에 따라 커진다
yr = 2 * xr + rng.normal(size=20000) * sd
u = rng.uniform(size=20000)
past = np.sort(2 * u + rng.normal(size=20000) * (0.1 + 0.5 * u))   # 과거 관측: 기후값 예측에 쓴다
s_pool = np.sqrt(np.mean(sd ** 2))
forecasts = {   # 이름: (PIT 값 (N,), 90% 구간 하한 (N,), 상한 (N,))
    "기후값": (np.searchsorted(past, yr) / len(past), *np.quantile(past, [0.05, 0.95])),
    "등분산": (norm.cdf(yr, 2 * xr, s_pool), *norm.ppf([[0.05], [0.95]], 2 * xr, s_pool)),
    "이분산": (norm.cdf(yr, 2 * xr, sd), *norm.ppf([[0.05], [0.95]], 2 * xr, sd)),
}
print("예측    90%포함 평균폭  포함(x<1/3) 포함(x>2/3)  PIT 5구간 비율")
for name, (pit, lo, hi) in forecasts.items():
    inside = (lo < yr) & (yr < hi)
    hist = np.histogram(pit, bins=5, range=(0, 1))[0] / len(yr)
    print(f"{name:6s}  {inside.mean():.3f}  {np.mean(hi - lo):.3f}     {inside[xr < 1/3].mean():.3f}"
          f"       {inside[xr > 2/3].mean():.3f}     {np.round(hist, 2)}")
for name, s in (("등분산", s_pool), ("이분산", sd)):
    print(f"{name}  NLL {np.mean(-norm.logpdf(yr, 2 * xr, s)):.3f}  CRPS {np.mean(crps_gauss(yr, 2 * xr, s)):.3f}")

# %% cell 4 — reliability diagram의 수치, ECE, temperature scaling
sig = lambda t: 1 / (1 + np.exp(-t))
z = rng.normal(0, 1.5, size=7000)              # (N,) 참 로짓
yb = (rng.uniform(size=7000) < sig(z)).astype(float)
logit = 2.5 * z                                # 과신하는 모델: 로짓을 2.5배로 낸다
cal, te = slice(0, 2000), slice(2000, None)    # 보정용 2000개, 평가용 5000개

def scores(q, y, bins=10):
    b = np.minimum((q * bins).astype(int), bins - 1)
    ece = sum(abs(y[b == k].mean() - q[b == k].mean()) * np.mean(b == k) for k in range(bins) if np.any(b == k))
    nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return f"정확도 {np.mean((q > 0.5) == y):.3f}  ECE {ece:.3f}  Brier {np.mean((q - y) ** 2):.3f}  NLL {nll:.3f}"

Ts = np.linspace(0.5, 5, 451)
nll_cal = [-np.mean(yb[cal] * np.log(sig(logit[cal] / T)) + (1 - yb[cal]) * np.log(1 - sig(logit[cal] / T))) for T in Ts]
T_best = Ts[np.argmin(nll_cal)]
print("보정 전     ", scores(sig(logit[te]), yb[te]))
print(f"T = {T_best:.2f}    ", scores(sig(logit[te] / T_best), yb[te]))
