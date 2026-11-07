"""13.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/bayesian-inference/conjugate-models/cells.py
"""
# %% cell 1 — Gamma–Poisson: 계수율의 사후분포
import numpy as np
from scipy import stats

counts = np.array([3, 12, 0, 41])                # 측정마다 센 계수 (12.2장 연습문제와 같은 데이터)
exposure_s = np.array([1.0, 5.0, 0.5, 20.0])     # 노출 시간 [s]
a0, b0 = 2.0, 1.0                                # 사전 Gamma(a0, b0): 평균 a0/b0 = 2 /s
a, b = a0 + counts.sum(), b0 + exposure_s.sum()  # 사후 Gamma(a0 + Σy, b0 + Σt)

rate = np.linspace(1e-4, 6, 60001)               # 계수율 격자 [1/s], (G,)
logpost = stats.gamma.logpdf(rate, a0, scale=1 / b0) \
    + stats.poisson.logpmf(counts[:, None], rate * exposure_s[:, None]).sum(axis=0)   # (G,)
grid = np.exp(logpost - logpost.max())
grid /= grid.sum() * (rate[1] - rate[0])         # 격자에서 정규화한 사후, (G,)
closed = stats.gamma.pdf(rate, a, scale=1 / b)   # 닫힌 형태의 사후, (G,)

lo, hi = stats.gamma.ppf([0.025, 0.975], a, scale=1 / b)
print(f"사후 Gamma({a:.0f}, {b:.1f})  평균 {a / b:.3f} /s  95% 구간 [{lo:.3f}, {hi:.3f}] /s")
print(f"격자 사후와 닫힌 형태의 최대 차이 < 1e-6: {np.max(np.abs(grid - closed)) < 1e-6}")
print(f"MLE Σy/Σt = {counts.sum() / exposure_s.sum():.3f} /s")

# %% cell 2 — Normal–Normal: 정밀도로 가중한 평균
sigma_c, mu0, tau0 = 0.3, 0.0, 0.25              # 잡음 [°C], 사전 N(mu0, tau0²) — 13.1장 센서 보정
rng = np.random.default_rng(0)
y = 0.4 + sigma_c * rng.normal(size=30)          # 참 bias 0.4 °C, (30,)

for n in (1, 3, 10, 30):
    prec = 1 / tau0**2 + n / sigma_c**2          # 사후 정밀도 = 사전 정밀도 + 데이터 정밀도
    w = (n / sigma_c**2) / prec                  # 표본평균에 주는 가중치
    mu_n = w * y[:n].mean() + (1 - w) * mu0      # 사후평균
    print(f"n = {n:2d}  표본평균 {y[:n].mean():+.3f}  가중치 {w:.3f}  사후 N({mu_n:+.3f}, {1 / np.sqrt(prec):.3f}²) °C")

# %% cell 3 — 베이지안 선형회귀: 데이터가 없는 곳에서 불확실성이 커진다
rng = np.random.default_rng(1)
x = np.r_[rng.uniform(-3, -1, 15), rng.uniform(1.5, 3, 15)]     # 가운데가 빈 입력, (N,)
y = np.sin(x) + 0.1 * rng.normal(size=30)                        # (N,)
sigma = 0.1                                                      # 관측 잡음 표준편차 (알려져 있다고 둔다)
alpha = 1.0                                                      # 가중치 사전 N(0, I/alpha)

def features(x):                                                 # 다항식 특징, (N, 6)
    return np.vander(x / 3, 6, increasing=True)

Phi = features(x)
S = np.linalg.inv(alpha * np.eye(6) + Phi.T @ Phi / sigma**2)    # 사후 공분산, (6, 6)
m = S @ Phi.T @ y / sigma**2                                     # 사후평균, (6,)

for x0 in (-2.0, 0.25, 2.0, 4.0):
    phi = features(np.array([x0]))[0]                            # (6,)
    epi = np.sqrt(phi @ S @ phi)                                 # 가중치 불확실성에서 온 표준편차
    tot = np.sqrt(sigma**2 + epi**2)                             # 예측분포의 표준편차
    print(f"x = {x0:5.2f}  평균 {phi @ m:+.3f}  참값 {np.sin(x0):+.3f}  epistemic {epi:.3f}  전체 {tot:.3f}")

# %% cell 4 — evidence로 다항식 차수 고르기
rng = np.random.default_rng(3)
x = rng.uniform(-1, 1, 25)
y = 1.0 - 2.0 * x + 1.5 * x**3 + 0.2 * rng.normal(size=25)      # 참 함수는 3차, 잡음 0.2

def log_evidence(deg, alpha=0.1, sigma=0.2):                     # log p(y) = log N(y | 0, σ²I + ΦΦᵀ/α)
    Phi = np.vander(x, deg + 1, increasing=True)                 # (N, deg+1)
    C = sigma**2 * np.eye(len(x)) + Phi @ Phi.T / alpha          # (N, N)
    return stats.multivariate_normal(np.zeros(len(x)), C).logpdf(y)

for deg in range(0, 10):
    Phi = np.vander(x, deg + 1, increasing=True)
    rmse = np.sqrt(np.mean((Phi @ np.linalg.lstsq(Phi, y, rcond=None)[0] - y) ** 2))
    print(f"차수 {deg}  학습 RMSE {rmse:.3f}  log evidence {log_evidence(deg):7.2f}")

# %% cell 5 — empirical Bayes: evidence를 최대화해 잡음과 사전 강도를 추정
sig_grid = np.linspace(0.05, 0.6, 56)            # 잡음 표준편차 후보, (A,)
alp_grid = np.logspace(-3, 3, 25)                # 사전 정밀도 후보, (B,)
le = np.array([[log_evidence(3, al, sg) for al in alp_grid] for sg in sig_grid])   # (A, B)
i, j = np.unravel_index(np.argmax(le), le.shape)
print(f"evidence 최대: σ = {sig_grid[i]:.3f} (참값 0.2),  α = {alp_grid[j]:.3g}")
