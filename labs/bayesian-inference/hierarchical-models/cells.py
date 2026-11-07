"""13.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/bayesian-inference/hierarchical-models/cells.py
"""
# %% cell 1 — 세 가지 pooling (하이퍼파라미터를 안다고 둘 때)
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
n_j = np.array([1, 2, 3, 5, 8, 12, 20, 40])     # 셀마다 측정 수, (J,)
mu, tau, sigma = 2.0, 0.3, 0.6                  # 모집단 평균, 셀 간 표준편차, 측정 잡음 [%/100 cycles]
theta = mu + tau * rng.normal(size=8)           # 셀마다 참 열화율, (J,)
y = [theta[j] + sigma * rng.normal(size=n) for j, n in enumerate(n_j)]

ybar = np.array([v.mean() for v in y])          # no pooling: 셀마다 따로, (J,)
grand = np.concatenate(y).mean()                # complete pooling: 모두 하나로
se = sigma / np.sqrt(n_j)                       # 셀 평균의 표준오차, (J,)
B = se**2 / (se**2 + tau**2)                    # 수축 계수: 모집단 평균 쪽으로 끌리는 비율, (J,)
partial = B * mu + (1 - B) * ybar               # partial pooling, (J,)

print(" n_j   참값   no pool  partial  수축 B")
for j in range(8):
    print(f"{n_j[j]:4d}  {theta[j]:.3f}   {ybar[j]:.3f}    {partial[j]:.3f}    {B[j]:.2f}")
for name, est in (("no pooling", ybar), ("complete pooling", np.full(8, grand)), ("partial pooling", partial)):
    print(f"셀 열화율 RMSE  {np.sqrt(np.mean((est - theta) ** 2)):.3f}   {name}")

# %% cell 2 — 하이퍼파라미터까지 추론: (μ, τ)의 격자 사후분포
mus = np.linspace(1.0, 3.0, 201)                # (A,)
taus = np.linspace(0.005, 1.5, 300)             # (T,)
M, T = np.meshgrid(mus, taus, indexing="ij")    # (A, T)
# ȳ_j | μ, τ ~ N(μ, τ² + se_j²): θ_j를 적분한 주변우도. 사전: μ 균등, τ ~ half-Normal(0, 1)
loglik = stats.norm.logpdf(ybar[:, None, None], M, np.sqrt(T**2 + se[:, None, None] ** 2)).sum(axis=0)
logpost = loglik + stats.halfnorm.logpdf(T, scale=1.0)
w = np.exp(logpost - logpost.max())
w /= w.sum()                                    # 격자 위 사후확률, (A, T)

p_tau = w.sum(axis=0)                           # τ의 주변 사후, (T,)
lo, hi = taus[np.searchsorted(np.cumsum(p_tau), [0.05, 0.95])]
print(f"τ 사후평균 {np.sum(taus * p_tau):.3f}  90% 구간 [{lo:.3f}, {hi:.3f}]  (참값 {tau})")

Bj = se[:, None, None] ** 2 / (se[:, None, None] ** 2 + T**2)                  # (J, A, T)
cond_mean = Bj * M + (1 - Bj) * ybar[:, None, None]                           # E[θ_j | μ, τ, y]
cond_var = 1 / (1 / se[:, None, None] ** 2 + 1 / T**2)                        # Var[θ_j | μ, τ, y]
post_mean = np.sum(w * cond_mean, axis=(1, 2))                                # (J,)
post_sd = np.sqrt(np.sum(w * (cond_var + cond_mean**2), axis=(1, 2)) - post_mean**2)
for j in (0, 3, 7):
    print(f"n = {n_j[j]:2d}  no pool {ybar[j]:.3f} ± {se[j]:.3f}   계층 모델 {post_mean[j]:.3f} ± {post_sd[j]:.3f}   참값 {theta[j]:.3f}")

# %% cell 3 — 반복 실험: 그룹 크기별 오차
def eb_partial(ybar, se, taus=np.linspace(0.005, 1.5, 300)):
    """τ를 주변우도 최대화(μ는 가중평균으로 profile)로 정하고 partial pooling 추정값을 낸다."""
    v = taus[:, None] ** 2 + se**2                                  # ȳ_j의 주변분산, (T, J)
    m = np.sum(ybar / v, axis=1) / np.sum(1 / v, axis=1)            # τ마다 μ̂, (T,)
    ll = -0.5 * np.sum(np.log(v) + (ybar - m[:, None]) ** 2 / v, axis=1)   # 주변 로그우도, (T,)
    k = np.argmax(ll)
    B = se**2 / (se**2 + taus[k] ** 2)
    return B * m[k] + (1 - B) * ybar, taus[k]

err_np, err_pp, t_hats = [], [], []
for _ in range(1000):
    th = mu + tau * rng.normal(size=8)
    yb = th + se * rng.normal(size=8)            # 셀 평균만 있으면 충분하다(충분통계량)
    est, t_hat = eb_partial(yb, se)
    err_np.append((yb - th) ** 2)
    err_pp.append((est - th) ** 2)
    t_hats.append(t_hat)
rmse_np, rmse_pp = np.sqrt(np.mean(err_np, axis=0)), np.sqrt(np.mean(err_pp, axis=0))
print("n_j      " + " ".join(f"{n:6d}" for n in n_j))
print("no pool  " + " ".join(f"{r:6.3f}" for r in rmse_np))
print("partial  " + " ".join(f"{r:6.3f}" for r in rmse_pp))
print(f"추정한 τ가 0.01 미만인 비율 {np.mean(np.array(t_hats) < 0.01):.2f}")

# %% cell 4 — funnel: centered와 non-centered 매개변수화
J, s = 8, 1.5                                   # 그룹 8개, log τ ~ N(0, 1.5²), 데이터 없이 사전분포만
def logp_centered(z):                           # z = (log τ, θ_1..θ_J),  θ_j ~ N(0, τ²)
    return -0.5 * (z[0] / s) ** 2 - J * z[0] - 0.5 * np.sum(z[1:] ** 2) * np.exp(-2 * z[0])
def logp_noncentered(z):                        # z = (log τ, η_1..η_J),  θ_j = τ η_j,  η_j ~ N(0, 1)
    return -0.5 * (z[0] / s) ** 2 - 0.5 * np.sum(z[1:] ** 2)

for name, logp in (("centered", logp_centered), ("non-centered", logp_noncentered)):
    rng = np.random.default_rng(4)
    z, lp, acc, trace = np.zeros(J + 1), 0.0, 0, []
    for _ in range(20000):                      # random-walk Metropolis, 보폭 0.5 (13.5장)
        prop = z + 0.5 * rng.normal(size=J + 1)
        lp_new = logp(prop)
        if np.log(rng.random()) < lp_new - lp:
            z, lp, acc = prop, lp_new, acc + 1
        trace.append(z[0])
    lt = np.array(trace[2000:])                 # log τ의 샘플
    print(f"{name:12s}  수락률 {acc / 20000:.2f}  P(log τ < -2) {np.mean(lt < -2):.3f}  최솟값 {lt.min():6.2f}")
print(f"{'참값':12s}                P(log τ < -2) {stats.norm.cdf(-2 / s):.3f}")
