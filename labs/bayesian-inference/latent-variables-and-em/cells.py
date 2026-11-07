"""13.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/bayesian-inference/latent-variables-and-em/cells.py
"""
# %% cell 1 — 주변 로그우도와 책임도
import numpy as np
from scipy import stats
from scipy.special import logsumexp

rng = np.random.default_rng(0)
z = rng.random(500) < 0.3                        # 잠재변수: 어느 세포 집단인가 (관측되지 않음), (N,)
x = np.where(z, rng.normal(1.0, 0.5, 500), rng.normal(3.0, 0.8, 500))   # 로그 형광 세기, (N,)

def log_joint(x, pi, mu, sd):                    # log p(x_n, z_n = k) = log π_k + log N(x_n | μ_k, σ_k²), (N, K)
    return np.log(pi) + stats.norm.logpdf(x[:, None], mu, sd)

pi, mu, sd = np.array([0.3, 0.7]), np.array([1.0, 3.0]), np.array([0.5, 0.8])   # 참 파라미터
loglik = logsumexp(log_joint(x, pi, mu, sd), axis=1).sum()      # Σ_n log Σ_k p(x_n, z_n = k)
print(f"참 파라미터에서 주변 로그우도 log p(x) = {loglik:.2f}")

x0 = np.array([1.0, 1.8, 2.2, 3.0])              # 책임도를 볼 점들, (4,)
lj = log_joint(x0, pi, mu, sd)                   # (4, K)
resp = np.exp(lj - logsumexp(lj, axis=1, keepdims=True))        # p(z = k | x), (4, K)
for xi, r in zip(x0, resp):
    print(f"x = {xi:.1f}   p(집단 1 | x) = {r[0]:.3f}   p(집단 2 | x) = {r[1]:.3f}")

# %% cell 2 — EM 알고리즘과 로그우도의 단조 증가
def em(x, pi, mu, sd, n_iter=100):
    lls = []
    for _ in range(n_iter):
        lj = log_joint(x, pi, mu, sd)                            # (N, K)
        lls.append(logsumexp(lj, axis=1).sum())                  # 현재 파라미터의 로그우도
        r = np.exp(lj - logsumexp(lj, axis=1, keepdims=True))    # E-step: 책임도, (N, K)
        Nk = r.sum(axis=0)                                       # 성분마다 "유효 개수", (K,)
        pi = Nk / len(x)                                         # M-step: 가중 MLE
        mu = (r * x[:, None]).sum(axis=0) / Nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / Nk)
    return pi, mu, sd, np.array(lls)

pi_h, mu_h, sd_h, lls = em(x, np.array([0.5, 0.5]), np.array([0.0, 1.0]), np.array([1.0, 1.0]))
for t in (0, 1, 2, 5, 10, 20, 50, 99):
    print(f"반복 {t:2d}  로그우도 {lls[t]:9.2f}")
print(f"로그우도가 줄어든 반복의 수: {np.sum(np.diff(lls) < -1e-9)}")
print(f"추정  π = {np.round(pi_h, 3)}  μ = {np.round(mu_h, 3)}  σ = {np.round(sd_h, 3)}")

# %% cell 3 — ELBO 분해: log p(x) = ELBO(q) + KL(q || p(z | x))
def elbo_and_kl(x, q, pi, mu, sd):               # q: 임의의 분포 q(z_n), (N, K)
    lj = log_joint(x, pi, mu, sd)
    log_post = lj - logsumexp(lj, axis=1, keepdims=True)         # log p(z_n | x_n), (N, K)
    elbo = np.sum(q * (lj - np.log(q)))
    kl = np.sum(q * (np.log(q) - log_post))
    return elbo, kl, logsumexp(lj, axis=1).sum()

theta0 = (np.array([0.5, 0.5]), np.array([0.0, 1.0]), np.array([1.0, 1.0]))   # EM의 초기 파라미터
q_uniform = np.full((500, 2), 0.5)
lj0 = log_joint(x, *theta0)
q_post = np.exp(lj0 - logsumexp(lj0, axis=1, keepdims=True))  # E-step이 고르는 q
elbo, kl, ll = elbo_and_kl(x, q_uniform, *theta0)
print(f"log p(x) = {ll:.2f}")
print(f"q = 균등:  ELBO {elbo:.2f}  KL {kl:.2f}  ELBO + KL = log p(x): {np.isclose(elbo + kl, ll)}")
elbo, kl, ll = elbo_and_kl(x, q_post, *theta0)
print(f"q = 사후:  ELBO {elbo:.2f}  KL < 1e-8: {abs(kl) < 1e-8}")

# %% cell 4 — 초기값에 따른 국소해
rng = np.random.default_rng(1)
x3 = np.concatenate([rng.normal(-3, 0.6, 150), rng.normal(0, 0.6, 150), rng.normal(0.9, 0.6, 150)])   # (N,)
finals = []
for seed in range(10):
    r = np.random.default_rng(seed)
    mu0 = r.choice(x3, size=3, replace=False)                    # 데이터 점 3개로 초기화
    p, m, s, ll = em(x3, np.full(3, 1 / 3), mu0, np.full(3, 1.0), n_iter=300)
    finals.append(ll[-1])
    print(f"초기 μ = {np.round(np.sort(mu0), 2)}  →  μ = {np.round(np.sort(m), 2)}  로그우도 {ll[-1]:.2f}")
print(f"최종 로그우도의 범위: {min(finals):.2f} ~ {max(finals):.2f}")

# %% cell 5 — 결측 메커니즘과 complete-case 분석
rng = np.random.default_rng(2)
n = 20000
u = rng.normal(size=n)                           # 늘 관측되는 공변량 (예: 공정 온도), (n,)
y = 2.0 + u + 0.5 * rng.normal(size=n)           # 가끔 빠지는 측정값, 참 평균 2.0, (n,)
masks = {                                        # True = 관측됨
    "MCAR": rng.random(n) < 0.6,                                  # 무작위로 40% 누락
    "MAR":  rng.random(n) < 1 / (1 + np.exp(2 * u)),              # u가 클수록 누락 (u는 관측됨)
    "MNAR": y < 2.8,                                              # 센서 포화: y 자체가 크면 누락
}
for name, obs in masks.items():
    b1, b0 = np.polyfit(u[obs], y[obs], 1)                       # 관측된 행으로 y ~ u 회귀
    print(f"{name:4s}  관측 비율 {obs.mean():.2f}  complete-case 평균 {y[obs].mean():.3f}"
          f"  회귀로 채운 평균 {np.mean(b0 + b1 * u):.3f}   (참값 2.000)")
