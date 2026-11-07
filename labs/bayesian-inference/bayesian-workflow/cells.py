"""13.8 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/bayesian-inference/bayesian-workflow/cells.py
"""
# %% cell 1 — 사전 예측 점검
import numpy as np

rng = np.random.default_rng(0)
for name, mu0, sd0 in [("기본 N(0, 10^2)", 0.0, 10.0), ("약한 정보 N(log 30, 1^2)", np.log(30), 1.0)]:
    lam = np.exp(rng.normal(mu0, sd0, size=4000))           # (4000,) 사전에서 뽑은 평균 집락 수
    q = np.quantile(lam, [0.05, 0.5, 0.95])
    print(f"{name:<24} lambda 5%/50%/95%: {q[0]:9.2g} {q[1]:9.2g} {q[2]:9.2g}"
          f"   1 미만 {np.mean(lam < 1):.2f}  1000 초과 {np.mean(lam > 1000):.2f}")

# %% cell 2 — 적합과 사후 예측 점검 (Poisson)
from scipy.special import gammaln, logsumexp

y = np.random.default_rng(1).negative_binomial(5, 5 / (5 + 30), size=40)   # (40,) 평판 40장의 집락 수
print(f"관측: 평균 {y.mean():.1f}  분산/평균 {y.var() / y.mean():.2f}  최댓값 {y.max()}")

th = np.linspace(np.log(5), np.log(100), 2001)              # (G,) log lambda 격자
log_post = (y.sum() * th - y.size * np.exp(th) - gammaln(y + 1).sum()
            - 0.5 * (th - np.log(30)) ** 2)                 # 사전 N(log 30, 1)
w = np.exp(log_post - logsumexp(log_post))                  # (G,) 격자 사후 가중치

def ppc(draw_rep, stat, n_rep=4000):                        # 사후 예측 p-value
    t_rep = np.array([stat(draw_rep()) for _ in range(n_rep)])
    return np.mean(t_rep >= stat(y)), np.quantile(t_rep, [0.05, 0.95])

disp = lambda d: d.var() / d.mean()                         # 검정 통계량: 분산/평균
rng = np.random.default_rng(2)
draw_pois = lambda: rng.poisson(np.exp(rng.choice(th, p=w)), size=y.size)
p, band = ppc(draw_pois, disp)
print(f"Poisson: 복제 데이터의 분산/평균 90% 범위 [{band[0]:.2f}, {band[1]:.2f}]  사후 예측 p = {p:.3f}")
p, band = ppc(draw_pois, np.mean)
print(f"Poisson: 복제 데이터의 평균 90% 범위 [{band[0]:.1f}, {band[1]:.1f}]  사후 예측 p = {p:.3f}")

# %% cell 3 — 모델 개선: 음이항 모델
def nb_logpmf(k, mu, phi):                                  # 평균 mu, 분산 mu + mu^2 / phi
    return (gammaln(k + phi) - gammaln(phi) - gammaln(k + 1)
            + phi * np.log(phi / (phi + mu)) + k * np.log(mu / (phi + mu)))

lm, lp = np.meshgrid(np.linspace(np.log(15), np.log(60), 151),
                     np.linspace(np.log(0.5), np.log(100), 151), indexing="ij")   # (151, 151)
ll_nb = nb_logpmf(y[:, None, None], np.exp(lm), np.exp(lp))         # (40, 151, 151) 점별 로그우도
log_post_nb = ll_nb.sum(0) - 0.5 * (lm - np.log(30)) ** 2 - 0.5 * (lp - np.log(10)) ** 2 / 1.5**2
w_nb = np.exp(log_post_nb - logsumexp(log_post_nb)).ravel()        # (151*151,)
mu_g, phi_g = np.exp(lm).ravel(), np.exp(lp).ravel()
print(f"음이항 사후 평균: mu {w_nb @ mu_g:.1f}  phi {w_nb @ phi_g:.1f}")

def draw_nb():
    i = rng.choice(w_nb.size, p=w_nb)
    return rng.negative_binomial(phi_g[i], phi_g[i] / (phi_g[i] + mu_g[i]), size=y.size)
p, band = ppc(draw_nb, disp)
print(f"음이항: 복제 데이터의 분산/평균 90% 범위 [{band[0]:.2f}, {band[1]:.2f}]  사후 예측 p = {p:.3f}")

# %% cell 4 — 시뮬레이션 기반 보정(SBC)
from scipy.stats import chisquare

def sbc(post_shape_scale, n_sims=1000, L=99, a=2.0, b=0.1, n=20):
    rng = np.random.default_rng(3)
    ranks = np.empty(n_sims, dtype=int)
    for s in range(n_sims):
        lam = rng.gamma(a, 1 / b)                           # 1) 사전에서 참값
        ys = rng.poisson(lam, size=n)                       # 2) 데이터 시뮬레이션
        shape = (a + ys.sum()) * post_shape_scale           # 3) 사후 표본 (Gamma 켤레)
        draws = rng.gamma(shape, (a + ys.sum()) / shape / (b + n), size=L)
        ranks[s] = np.sum(draws < lam)                      # 4) 참값의 순위 (0..L)
    counts = np.bincount(ranks // 10, minlength=10)         # 순위 0..99를 10칸으로
    return counts, chisquare(counts).pvalue

for name, scale in [("올바른 사후", 1.0), ("과신한 사후(분산 1/4)", 4.0)]:
    counts, pval = sbc(scale)
    print(f"{name:<20} 순위 히스토그램 {counts}  균일성 p = {pval:.3f}")

# %% cell 5 — 모델 비교: 정확한 LOO와 WAIC
def loo_waic(ll, log_w):                                    # ll: (N, G) 점별 로그우도, log_w: (G,) 로그 사후 가중치
    elpd_loo = -logsumexp(-ll + log_w, axis=1)              # log p(y_i | y_-i) = -log E[1 / p(y_i | th)]
    lppd = logsumexp(ll + log_w, axis=1)                    # log E[p(y_i | th)]
    w = np.exp(log_w)
    p_waic = (ll**2) @ w - (ll @ w) ** 2                    # 사후분산 = 유효 파라미터 수
    return elpd_loo, lppd - p_waic

ll_pois = y[:, None] * th[None, :] - np.exp(th)[None, :] - gammaln(y + 1)[:, None]   # (40, G)
loo_p, waic_p = loo_waic(ll_pois, log_post - logsumexp(log_post))
loo_n, waic_n = loo_waic(ll_nb.reshape(y.size, -1), (log_post_nb - logsumexp(log_post_nb)).ravel())
d = loo_n - loo_p                                           # (40,) 점별 차이
print(f"elpd_loo  Poisson {loo_p.sum():7.1f}   음이항 {loo_n.sum():7.1f}")
print(f"elpd_waic Poisson {waic_p.sum():7.1f}   음이항 {waic_n.sum():7.1f}")
print(f"차이(음이항 - Poisson) {d.sum():.1f} ± {np.sqrt(y.size * d.var()):.1f} (표준오차)")
