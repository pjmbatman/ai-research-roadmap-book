"""13.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/bayesian-inference/bayesian-thinking/cells.py
"""
# %% cell 1 — 격자에서 사전·우도·evidence·사후 계산
import numpy as np
from scipy import stats

n, k = 20, 7                                     # 20번 던져 앞면 7번 (12.2장과 같은 데이터)
theta = np.linspace(0, 1, 2001)                  # 파라미터 격자, (G,)
dtheta = theta[1] - theta[0]

prior = stats.beta.pdf(theta, 2, 2)              # p(θ) = Beta(2, 2), (G,)
lik = stats.binom.pmf(k, n, theta)               # p(D | θ), (G,)
evidence = np.sum(lik * prior) * dtheta          # p(D) = ∫ p(D | θ) p(θ) dθ
post = lik * prior / evidence                    # p(θ | D), (G,)

print(f"evidence p(D)            = {evidence:.5f}")
print(f"사후분포의 적분           = {np.sum(post) * dtheta:.4f}")
print(f"격자 사후와 Beta(9, 15)의 최대 차이 < 1e-10: {np.max(np.abs(post - stats.beta.pdf(theta, 9, 15))) < 1e-10}")
print(f"P(다음 던지기가 앞면 | D) = {np.sum(theta * post) * dtheta:.4f}   (9/24 = {9 / 24:.4f})")

# %% cell 2 — 사후예측분포와 plug-in 예측
m = 20                                           # 앞으로 20번 더 던진다
ks = np.arange(m + 1)                            # 가능한 앞면 수, (m+1,)
plug = stats.binom.pmf(ks, m, k / n)             # MLE θ̂ = 0.35를 꽂은 예측, (m+1,)
pred = stats.betabinom.pmf(ks, m, 9, 15)         # 사후예측 ∫ Binom(m, θ) p(θ | D) dθ, (m+1,)

rng = np.random.default_rng(0)                   # 같은 분포를 두 단계 샘플링으로 확인
sim = rng.binomial(m, rng.beta(9, 15, size=200_000))   # θ ~ 사후, 그 θ로 20번 던지기

for name, p in (("plug-in Binom(20, 0.35)", plug), ("사후예측 Beta-Binomial", pred)):
    mean = np.sum(ks * p)
    sd = np.sqrt(np.sum((ks - mean) ** 2 * p))
    lo, hi = ks[np.searchsorted(np.cumsum(p), [0.05, 0.95])]
    print(f"평균 {mean:.2f}  표준편차 {sd:.2f}  90% 범위 [{lo:2d}, {hi:2d}]   {name}")
print(f"평균 {sim.mean():.2f}  표준편차 {sim.std():.2f}                     두 단계 샘플링")

# %% cell 3 — 순차적 갱신
rng = np.random.default_rng(1)
tosses = (rng.random(50) < 0.35).astype(int)     # 참 θ = 0.35에서 50번, (50,)

a, b = 2, 2                                      # 사전 Beta(2, 2)
for i, t in enumerate(tosses, 1):
    a, b = a + t, b + (1 - t)                    # 오늘의 사후가 내일의 사전
    if i in (1, 5, 20, 50):
        lo, hi = stats.beta.ppf([0.025, 0.975], a, b)
        print(f"n = {i:2d}  사후 Beta({a:2d}, {b:2d})  평균 {a / (a + b):.3f}  95% 구간 [{lo:.3f}, {hi:.3f}]")

h = tosses.sum()
print(f"한 번에 갱신                 사후 Beta({2 + h:2d}, {2 + 50 - h:2d})")

# %% cell 4 — MLE·MAP·사후평균, 그리고 MAP의 매개변수화 의존성
a, b = 2 + 3, 2 + 0                              # 사전 Beta(2, 2), 3번 중 3번 앞면 → Beta(5, 2)
print(f"MLE       k/n           = {3 / 3:.3f}")
print(f"MAP       (a-1)/(a+b-2) = {(a - 1) / (a + b - 2):.3f}")
print(f"사후평균  a/(a+b)       = {a / (a + b):.3f}")
print(f"사후중앙값              = {stats.beta.median(a, b):.3f}")

# 같은 사후분포를 log-odds φ = log(θ/(1-θ))로 나타내면 밀도에 Jacobian dθ/dφ = θ(1-θ)가 곱해진다
phi = np.linspace(-6, 8, 140001)                 # (G,)
th = 1 / (1 + np.exp(-phi))                      # φ에 대응하는 θ, (G,)
dens_phi = stats.beta.pdf(th, a, b) * th * (1 - th)    # p(φ | D), (G,)
cdf_phi = np.cumsum(dens_phi) / dens_phi.sum()
print(f"φ 공간의 MAP를 θ로 되돌린 값      = {th[np.argmax(dens_phi)]:.3f}")
print(f"φ 공간의 중앙값을 θ로 되돌린 값   = {th[np.searchsorted(cdf_phi, 0.5)]:.3f}")

# %% cell 5 — 신뢰구간과 신용구간은 서로 다른 질문에 답한다
rng = np.random.default_rng(2)
n, R = 20, 20000                                 # 20번 던지는 실험을 R번 반복

ks = np.arange(n + 1)                            # 가능한 앞면 수마다 구간을 한 번씩 계산, (n+1,)
cp = (np.nan_to_num(stats.beta.ppf(0.025, ks, n - ks + 1), nan=0.0),    # Clopper-Pearson
      np.nan_to_num(stats.beta.ppf(0.975, ks + 1, n - ks), nan=1.0))    # 빈도주의 정확 신뢰구간
cr = stats.beta.ppf([[0.025], [0.975]], 1 + ks, 1 + n - ks)  # 균등 사전 Beta(1, 1)의 신용구간

def report(label, th, k):
    for name, (lo, hi) in (("Clopper-Pearson 신뢰구간", cp), ("신용구간", cr)):
        print(f"{label}  포함률 {np.mean((lo[k] <= th) & (th <= hi[k])):.3f}   {name}")

report("θ = 0.05 고정  ", 0.05, rng.binomial(n, 0.05, size=R))
th = rng.beta(1, 1, size=R)                      # θ를 실험마다 사전분포에서 새로 뽑는다, (R,)
report("θ ~ Beta(1, 1)", th, rng.binomial(n, th))
