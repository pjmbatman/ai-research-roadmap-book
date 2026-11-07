"""27.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/earth-and-climate/climate-evaluation/cells.py
"""
# %% cell 1 — 날카로운 예보와 흐릿한 예보: RMSE·ACC
import numpy as np

rng = np.random.default_rng(0)
N, L = 500, 128                                   # 사례 수, 위도권 한 바퀴의 격자점 수
k = np.arange(1, L // 2)                          # (63,) 동서 파수
amp = k ** -1.5                                   # 에너지 스펙트럼 E(k) ∝ k^-3
rho = np.exp(-k / 6)                              # 파수별 예측 가능성: 큰 규모일수록 잘 맞는다

def field(coef):                                  # coef: (N, 63) 복소 계수 → (N, L) 실수 장
    full = np.zeros((coef.shape[0], L // 2 + 1), complex); full[:, 1:L // 2] = coef
    return np.fft.irfft(full, n=L) * L

cplx = lambda: amp * (rng.normal(size=(N, k.size)) + 1j * rng.normal(size=(N, k.size))) / np.sqrt(2)
c_a, c_b, c_b2 = cplx(), cplx(), cplx()           # 예측 가능한 부분, 예측할 수 없는 부분과 그 다른 실현
truth = field(rho * c_a + np.sqrt(1 - rho**2) * c_b)
sharp = field(rho * c_a + np.sqrt(1 - rho**2) * c_b2)    # 실현 가능한 장 하나(스펙트럼이 정답과 같다)
blurry = field(rho * c_a)                         # 조건부 평균(MSE 최적): 작은 규모가 사라진다

acc = lambda f: np.mean((f * truth).sum(1) / np.sqrt((f**2).sum(1) * (truth**2).sum(1)))
for name, f in [("날카로운 예보", sharp), ("흐릿한 예보", blurry)]:
    print(f"{name}: RMSE {np.sqrt(((f - truth)**2).mean()):.3f}  ACC {acc(f):.3f}  표준편차 {f.std():.3f}")
print(f"정답 표준편차 {truth.std():.3f}")

# %% cell 2 — 앙상블의 CRPS, spread–skill, rank histogram
M = 10
def ensemble(scale):                              # scale < 1이면 예측할 수 없는 부분을 과소 표현
    return np.stack([field(rho * c_a + scale * np.sqrt(1 - rho**2) * cplx()) for _ in range(M)])  # (M, N, L)

def crps(ens):                                    # 격자점별 표본 CRPS의 평균
    xs, w = np.sort(ens, 0), 2 * np.arange(1, len(ens) + 1) - len(ens) - 1
    pair = 2 * (w[:, None, None] * xs).sum(0) / len(ens) ** 2
    return (np.abs(ens - truth).mean(0) - pair / 2).mean()

print(f"흐릿한 결정론 예보 CRPS(=MAE) {np.abs(blurry - truth).mean():.3f}")
print(f"날카로운 단일 예보 CRPS(=MAE) {np.abs(sharp - truth).mean():.3f}")
for name, s in [("보정된 앙상블", 1.0), ("퍼짐 부족 앙상블", 0.5)]:
    ens = ensemble(s)
    rm = np.sqrt(((ens.mean(0) - truth) ** 2).mean())
    sp = np.sqrt(ens.var(0, ddof=1).mean() * (M + 1) / M)          # 유한 앙상블 보정
    rank = (ens < truth).sum(0).ravel()                             # 정답의 순위 0..M
    hist = np.bincount(rank, minlength=M + 1) / rank.size
    print(f"{name}: CRPS {crps(ens):.3f}  spread/RMSE {sp / rm:.2f}  "
          f"순위 양 끝 비율 {hist[0] + hist[-1]:.2f} (평평하면 {2 / (M + 1):.2f})")

# %% cell 3 — 에너지 스펙트럼으로 흐릿함 진단
def power(f):                                     # (N, L) → (L//2+1,) 파수별 평균 에너지
    return (np.abs(np.fft.rfft(f, axis=1)) ** 2).mean(0) / L**2

ks = [1, 2, 4, 8, 16, 32]
print("파수 k          " + "".join(f"{kk:7d}" for kk in ks))
for name, f in [("날카로운 예보", sharp), ("흐릿한 예보", blurry), ("보정된 앙상블 평균", ensemble(1.0).mean(0))]:
    print(f"{name:10s}" + "".join(f"{v:7.2f}" for v in power(f)[ks] / power(truth)[ks]))

# %% cell 4 — 연 최댓값의 GEV와 재현 수준
from scipy import stats

def annual_max(g, n_year, warming_c_per_yr=0.0):  # 여름 92일의 일 최고기온(AR(1) 편차)에서 해마다 최댓값
    a = np.zeros(n_year); mx = np.full(n_year, -np.inf)
    for _ in range(92):
        a = 0.7 * a + g.normal(0, 3.0 * np.sqrt(1 - 0.7**2), n_year)
        mx = np.maximum(mx, a)
    return 30.0 + mx + warming_c_per_yr * np.arange(n_year)    # (n_year,) [°C]

g = np.random.default_rng(1)
ref = annual_max(g, 200_000)                      # 아주 긴 정상 기후: '참' 재현 수준의 기준
print(f"참 100년 재현 수준 {np.quantile(ref, 1 - 1 / 100):.2f} °C")
for i in range(3):                                # 같은 기후의 서로 다른 30년 표본
    xi_neg, loc, scale = stats.genextreme.fit(annual_max(g, 30))   # scipy의 c는 -ξ
    rl = lambda T: stats.genextreme.ppf(1 - 1 / T, xi_neg, loc, scale)
    print(f"30년 표본 {i}: ξ = {-xi_neg:+.2f}  10년 {rl(10):.2f}  100년 {rl(100):.2f} °C")
level = np.quantile(ref, 1 - 1 / 100)
for warm in [0.5, 1.0, 2.0]:                      # 분포 전체가 평행이동한 미래 기후
    p = np.mean(ref + warm > level)
    print(f"평균 +{warm:.1f} °C: 과거 100년 수준을 넘을 확률 {p:.3f}/년 (재현 기간 {1 / p:.0f}년)")

# %% cell 5 — 추세의 유의성: 자기상관과 유효 표본 수
def trend_t(y):                                   # y: (R, n) → 기울기, 순진한 t, 보정한 t, n_eff
    n = y.shape[1]; t = np.arange(n) - (n - 1) / 2
    b = (y * t).sum(1) / (t**2).sum()
    res = y - y.mean(1, keepdims=True) - b[:, None] * t
    se = np.sqrt((res**2).sum(1) / (n - 2) / (t**2).sum())
    r1 = (res[:, 1:] * res[:, :-1]).sum(1) / (res**2).sum(1)          # 잔차의 lag-1 자기상관
    n_eff = np.clip(n * (1 - r1) / (1 + r1), 3, n)
    return b, b / se, b / (se * np.sqrt((n - 2) / (n_eff - 2))), n_eff

g = np.random.default_rng(4)
R, n, phi = 4000, 50, 0.6                         # 추세가 없는 AR(1) 연평균 편차 4000개, 각 50년
y = np.zeros((R, n)); y[:, 0] = g.normal(0, 1, R)
for t in range(1, n):
    y[:, t] = phi * y[:, t - 1] + np.sqrt(1 - phi**2) * g.normal(0, 1, R)
b, t_naive, t_adj, n_eff = trend_t(y)
crit = stats.t.ppf(0.975, n - 2)
print(f"유효 표본 수 중앙값 {np.median(n_eff):.1f} / {n}")
print(f"추세 없음인데 '유의'(5%): 순진한 검정 {np.mean(np.abs(t_naive) > crit):.3f}"
      f"  보정한 검정 {np.mean(np.abs(t_adj) > stats.t.ppf(0.975, n_eff - 2)):.3f}")
