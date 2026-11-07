"""13.6 실습: 같은 사후분포를 MCMC·변분추론·Laplace 근사로 구해 모양과 불확실성을 비교한다.

모델: 희귀 사건 계수 y_i ~ Poisson(lambda), theta = log lambda ~ N(0, 1.5^2). 계수가 작아 사후분포가 비대칭이다.
실행: uv run --project labs python labs/bayesian-inference/variational-inference/compare_approximations.py
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln, logsumexp
from scipy.stats import norm

y = np.array([0, 1, 0, 2, 0, 0, 1, 0])                  # (8,) 관측 구간별 사건 수
N, S0 = y.size, 1.5


def log_joint(th):
    th = np.asarray(th, dtype=float)
    return np.sum(y) * th - N * np.exp(th) - gammaln(y + 1).sum() - 0.5 * th**2 / S0**2


def elbo(v):                                           # q = N(m, s^2), 닫힌 꼴(상수 제외)
    m, s = v[0], np.exp(v[1])
    return np.sum(y) * m - N * np.exp(m + s**2 / 2) - 0.5 * (m**2 + s**2) / S0**2 + np.log(s)


def summarize(name, lam):
    q5, q50, q95 = np.quantile(lam, [0.05, 0.5, 0.95])
    print(f"{name:<22} 평균 {lam.mean():.3f}  sd {lam.std():.3f}  5% {q5:6.3f}  중앙값 {q50:.3f}  95% {q95:.3f}")


rng = np.random.default_rng(0)

# 1) 기준: 수치 적분(격자)에서 뽑은 표본
grid = np.linspace(-6, 3, 20001)
w = np.exp(log_joint(grid) - logsumexp(log_joint(grid)))
summarize("격자(기준)", np.exp(rng.choice(grid, size=200000, p=w)))

# 2) MCMC: random-walk Metropolis, 체인 4개
chains = []
for c in range(4):
    th, lp, out = 0.0, log_joint(0.0), []
    for t in range(20000):
        prop = th + 0.8 * rng.normal()
        lp1 = log_joint(prop)
        if np.log(rng.random()) < lp1 - lp:
            th, lp = prop, lp1
        out.append(th)
    chains.append(out[5000:])
summarize("MCMC (MH, 4x15000)", np.exp(np.concatenate(chains)))

# 3) 변분추론: q(theta) = N(m, s^2), ELBO 최대화
v = minimize(lambda v: -elbo(v), [0.0, 0.0]).x
summarize("VI (theta 공간 가우시안)", np.exp(v[0] + np.exp(v[1]) * rng.normal(size=200000)))

# 4) Laplace 근사: theta = log lambda 공간
th_hat = minimize_scalar(lambda t: -log_joint(t), bounds=(-6, 3), method="bounded").x
H = N * np.exp(th_hat) + 1 / S0**2
summarize("Laplace (log lambda)", np.exp(th_hat + rng.normal(size=200000) / np.sqrt(H)))

# 5) Laplace 근사: lambda 공간(같은 사후를 lambda로 다시 쓴 밀도, 야코비안 포함)
def log_post_lam(lam):
    return log_joint(np.log(lam)) - np.log(lam)
lam_hat = minimize_scalar(lambda l: -log_post_lam(l), bounds=(1e-6, 10), method="bounded").x
h = 1e-5
H_lam = -(log_post_lam(lam_hat + h) - 2 * log_post_lam(lam_hat) + log_post_lam(lam_hat - h)) / h**2
lam_lap = lam_hat + rng.normal(size=200000) / np.sqrt(H_lam)
summarize("Laplace (lambda)", lam_lap)
print(f"Laplace (lambda)의 음수 표본 비율: {np.mean(lam_lap < 0):.3f}")
print(f"VI 파라미터: m {v[0]:.3f}, s {np.exp(v[1]):.3f}   Laplace: 최빈값 {th_hat:.3f}, sd {1 / np.sqrt(H):.3f}")
