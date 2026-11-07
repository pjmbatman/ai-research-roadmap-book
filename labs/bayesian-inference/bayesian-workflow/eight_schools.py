"""13.8 실습: eight schools 데이터로 베이지안 워크플로 한 사이클을 수행한다.

데이터: SAT 코칭 프로그램의 학교별 효과 추정치 y_j와 표준오차 sigma_j (Rubin, 1981).
모델: y_j ~ N(theta_j, sigma_j^2), theta_j ~ N(mu, tau^2), mu ~ N(0, 20^2), tau ~ (사전분포 비교)
theta를 적분한 주변 사후 p(mu, tau | y)는 2차원이라 격자로 정확히 계산하고, MCMC 결과와 비교한다.
실행: uv run --project labs python labs/bayesian-inference/bayesian-workflow/eight_schools.py
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import halfcauchy, halfnorm, norm

y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])        # (8,) 효과 추정치 [점]
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])  # (8,) 표준오차 [점]
J = y.size
rng = np.random.default_rng(0)

MU = np.linspace(-40, 60, 401)
TAU = np.linspace(0.01, 60, 600)
M, T = np.meshgrid(MU, TAU, indexing="ij")                         # (401, 600)
PRIORS = {"half-normal(0, 10)": halfnorm(scale=10).logpdf,
          "half-Cauchy(0, 25)": halfcauchy(scale=25).logpdf,
          "uniform(0, 60)": lambda t: np.zeros_like(t)}


def pointwise_ll(m, t):                                            # (J, ...) log N(y_j | mu, sigma_j^2 + tau^2)
    return norm.logpdf(y.reshape(-1, *[1] * np.ndim(m)), m, np.sqrt(sigma.reshape(-1, *[1] * np.ndim(m)) ** 2 + t**2))


def grid_posterior(log_prior_tau):
    lp = pointwise_ll(M, T).sum(0) + norm.logpdf(M, 0, 20) + log_prior_tau(T)
    return lp - logsumexp(lp)                                      # (401, 600) 로그 가중치


def draw(log_w, n):
    idx = rng.choice(log_w.size, size=n, p=np.exp(log_w).ravel())
    mu, tau = M.ravel()[idx], T.ravel()[idx]
    prec = 1 / sigma[:, None] ** 2 + 1 / tau**2                     # (J, n) theta_j | mu, tau, y
    mean = (y[:, None] / sigma[:, None] ** 2 + mu / tau**2) / prec
    theta = mean + rng.normal(size=(J, n)) / np.sqrt(prec)
    return mu, tau, theta


def q(x):
    a, b, c = np.quantile(x, [0.05, 0.5, 0.95])
    return f"{b:5.1f} [{a:5.1f}, {c:5.1f}]"


print("1) 사전 예측 점검: 학교 효과 y_rep의 90% 범위 [점]")
for name, mu_sd, tau_draw in [("기본(넓은) 사전: mu~N(0,1000^2), tau~half-normal(0,1000)", 1000, lambda n: halfnorm(scale=1000).rvs(n, random_state=rng)),
                              ("약한 정보 사전: mu~N(0,20^2), tau~half-normal(0,10)", 20, lambda n: halfnorm(scale=10).rvs(n, random_state=rng))]:
    n = 20000
    theta = rng.normal(0, mu_sd, n) + tau_draw(n) * rng.normal(size=n)
    y_rep = theta + sigma[rng.integers(0, J, n)] * rng.normal(size=n)
    lo, hi = np.quantile(y_rep, [0.05, 0.95])
    print(f"   {name:<55} [{lo:8.1f}, {hi:8.1f}]")

log_w = grid_posterior(PRIORS["half-normal(0, 10)"])
mu_s, tau_s, theta_s = draw(log_w, 20000)
print("\n2) 적합(격자, 정확): 중앙값 [90% 구간]")
print(f"   mu {q(mu_s)}   tau {q(tau_s)}   theta_1 {q(theta_s[0])}")

# 3) 진단: 같은 주변 사후를 random-walk Metropolis로 샘플링해 격자와 비교
def log_target(v):                                                 # v = (mu, log tau), 야코비안 포함
    s2 = sigma**2 + np.exp(2 * v[1])
    ll = -0.5 * np.sum(np.log(2 * np.pi * s2) + (y - v[0]) ** 2 / s2)
    return ll - 0.5 * v[0] ** 2 / 20**2 - 0.5 * np.exp(2 * v[1]) / 10**2 + v[1]


def run_mh(n_iter):
    chains = []
    for start in ([-30, 3.5], [50, 3.5], [-30, -3], [50, -3]):     # 흩어진 시작점
        v, lp, out = np.array(start, float), log_target(np.array(start, float)), []
        for _ in range(n_iter):
            prop = v + rng.normal(size=2) * [8.0, 1.0]
            lp1 = log_target(prop)
            if np.log(rng.random()) < lp1 - lp:
                v, lp = prop, lp1
            out.append(v.copy())
        chains.append(np.array(out[n_iter // 2:]))                 # 앞 절반은 워밍업
    return np.array(chains)                                        # (4, n_iter / 2, 2)


def split_rhat(c):
    h = c.shape[1] // 2
    c = np.vstack([c[:, :h], c[:, h:2 * h]])
    n = c.shape[1]
    W, B = c.var(axis=1, ddof=1).mean(), n * c.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * W + B / n) / W)


def ess(c):
    Mc, N = c.shape
    x = c - c.mean(axis=1, keepdims=True)
    f = np.fft.rfft(x, 2 * N, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :N]
    rho = (ac / ac[:, :1]).mean(axis=0)
    tau, k = -1.0, 0
    while 2 * k + 1 < N and rho[2 * k] + rho[2 * k + 1] > 0:
        tau += 2 * (rho[2 * k] + rho[2 * k + 1])
        k += 1
    return Mc * N / tau


for n_iter in (4000, 40000):
    chains = run_mh(n_iter)
    print(f"\n3) 진단(MH 체인 4개, 반복 {n_iter}, 뒤 절반 사용):")
    for i, name in enumerate(("mu", "log tau")):
        print(f"   {name:<8} R-hat {split_rhat(chains[..., i]):.3f}  ESS {ess(chains[..., i]):5.0f}")
    print(f"   MH  mu {q(chains[..., 0].ravel())}   tau {q(np.exp(chains[..., 1]).ravel())}")

print("\n4) 사후 예측 점검: 사후 예측 p-value")
y_rep = theta_s + sigma[:, None] * rng.normal(size=theta_s.shape)   # (J, n)
for name, f in [("최댓값", np.max), ("최솟값", np.min), ("표준편차", np.std)]:
    t_rep = f(y_rep, axis=0)
    print(f"   {name:<6} 관측 {f(y):5.1f}   p = {np.mean(t_rep >= f(y)):.2f}")

print("\n5) 사전 민감도: tau의 사전분포에 따른 사후 요약")
for name, lpt in PRIORS.items():
    _, tau_k, theta_k = draw(grid_posterior(lpt), 20000)
    print(f"   {name:<20} tau {q(tau_k)}   theta_1 {q(theta_k[0])}")

print("\n6) 모델 비교: 학교 하나를 뺀 정확한 LOO (격자)")
ll = pointwise_ll(M, T).reshape(J, -1)                            # (J, G)
loo_h = -logsumexp(-ll + log_w.ravel(), axis=1)                    # 계층 모델
lp_pool = pointwise_ll(MU, np.zeros_like(MU)).sum(0) + norm.logpdf(MU, 0, 20)   # tau = 0: 완전 합동
lw_pool = lp_pool - logsumexp(lp_pool)
loo_p = -logsumexp(-pointwise_ll(MU, np.zeros_like(MU)) + lw_pool, axis=1)
d = loo_h - loo_p
print(f"   elpd_loo 계층 {loo_h.sum():6.2f}   완전 합동 {loo_p.sum():6.2f}   차이 {d.sum():5.2f} ± {np.sqrt(J * d.var()):.2f}")
