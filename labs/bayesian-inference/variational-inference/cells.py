"""13.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/bayesian-inference/variational-inference/cells.py
"""
# %% cell 1 — ELBO + KL = log evidence
import numpy as np
from scipy.special import gammaln, logsumexp

rng = np.random.default_rng(0)
y = rng.poisson(4.0, size=200)                  # (200,) 검출기 계수, 참 rate 4
N, S0 = y.size, 1.5                             # theta = log rate ~ N(0, 1.5^2)

def log_joint(th):                              # log p(y, theta), th: (K,)
    lik = np.sum(y) * th - N * np.exp(th) - gammaln(y + 1).sum()
    return lik - 0.5 * th**2 / S0**2 - 0.5 * np.log(2 * np.pi * S0**2)

def elbo(m, s):                                 # q = N(m, s^2), 모든 항이 닫힌 꼴
    e_lik = np.sum(y) * m - N * np.exp(m + s**2 / 2) - gammaln(y + 1).sum()
    e_prior = -0.5 * (m**2 + s**2) / S0**2 - 0.5 * np.log(2 * np.pi * S0**2)
    return e_lik + e_prior + 0.5 * np.log(2 * np.pi * np.e * s**2)

grid = np.linspace(-1.0, 3.0, 40001)            # (G,) theta 격자
dx = grid[1] - grid[0]
lj = log_joint(grid)
log_Z = logsumexp(lj) + np.log(dx)              # log p(y), 수치 적분
log_post = lj - log_Z                           # (G,) 정규화된 로그 사후밀도
post = np.exp(log_post)
for m, s in [(1.0, 0.2), (1.45, 0.1), (1.42, 0.035)]:
    log_q = -0.5 * ((grid - m) / s) ** 2 - np.log(s * np.sqrt(2 * np.pi))
    kl = np.sum(np.exp(log_q) * (log_q - log_post)) * dx    # KL(q || p)
    print(f"q=N({m}, {s}^2): ELBO {elbo(m, s):9.3f}  KL {kl:7.3f}  ELBO+KL {elbo(m, s) + kl:9.3f}")
print(f"log p(y) (수치 적분)          {log_Z:9.3f}")

# %% cell 2 — 확률적 변분추론: 재매개변수화 + mini-batch
def grad_log_joint(th, yb):                     # mini-batch로 추정한 d log p(y, th) / d th
    return (N / yb.size) * (np.sum(yb) - yb.size * np.exp(th)) - th / S0**2

m, log_s, B, lr = 0.0, np.log(0.5), 20, 5e-4
rng = np.random.default_rng(1)
hist = []
for t in range(4000):
    yb = rng.choice(y, size=B, replace=False)   # (B,) mini-batch
    eps = rng.normal(size=8)                    # (8,) 재매개변수화 잡음
    th = m + np.exp(log_s) * eps                # (8,) theta = m + s * eps
    g = grad_log_joint(th, yb)                  # (8,)
    g_m = g.mean()                              # d ELBO / d m 추정
    g_ls = (g * eps).mean() * np.exp(log_s) + 1.0   # d ELBO / d log s (+1은 엔트로피)
    m, log_s = m + lr * g_m, log_s + lr * g_ls
    hist.append((m, log_s))
m_avg, ls_avg = np.mean(hist[2000:], axis=0)    # 뒤 절반의 반복 평균
post_mean = np.sum(grid * post) * dx
post_sd = np.sqrt(np.sum((grid - post_mean) ** 2 * post) * dx)
print(f"SVI 마지막 반복 : 평균 {m:.4f}  sd {np.exp(log_s):.4f}")
print(f"SVI 반복 평균   : 평균 {m_avg:.4f}  sd {np.exp(ls_avg):.4f}")
print(f"참 사후(수치)   : 평균 {post_mean:.4f}  sd {post_sd:.4f}")

# %% cell 3 — Laplace 근사
from scipy.optimize import minimize_scalar

th_hat = minimize_scalar(lambda t: -log_joint(np.array([t]))[0], bounds=(0, 3), method="bounded",
                         options={"xatol": 1e-10}).x               # 사후 최빈값
H = N * np.exp(th_hat) + 1 / S0**2              # -d^2 log p / d theta^2 (닫힌 꼴)
log_Z_lap = log_joint(np.array([th_hat]))[0] + 0.5 * np.log(2 * np.pi / H)
print(f"Laplace : 최빈값 {th_hat:.4f}  sd {1 / np.sqrt(H):.4f}")
print(f"log p(y): Laplace {log_Z_lap:.3f}   수치 적분 {log_Z:.3f}")

# %% cell 4 — mean-field의 분산 과소평가
rho = 0.9
Sigma = np.array([[1.0, rho], [rho, 1.0]])      # (2, 2) 참 사후 공분산
Lam = np.linalg.inv(Sigma)                      # 정밀도 행렬
sd_reverse = 1 / np.sqrt(np.diag(Lam))          # argmin KL(q || p): 정밀도의 대각
sd_forward = np.sqrt(np.diag(Sigma))            # argmin KL(p || q): 주변분포의 분산
print("참 주변 sd          :", sd_forward)
print("mean-field (q||p) sd:", sd_reverse.round(3), "  sqrt(1 - rho^2) =", round(np.sqrt(1 - rho**2), 3))

# %% cell 5 — KL의 방향: mode-seeking과 mass-covering
from scipy.optimize import minimize

x = np.linspace(-8, 8, 4001)                    # (G,)
dx = x[1] - x[0]
log_n = lambda x, m, s: -0.5 * ((x - m) / s) ** 2 - np.log(s * np.sqrt(2 * np.pi))
p = 0.5 * np.exp(log_n(x, -2, 0.6)) + 0.5 * np.exp(log_n(x, 2, 0.6))   # 쌍봉 사후
def kl(a, b):                                   # 격자 위 KL(a || b)
    return np.sum(a * (np.log(a + 1e-300) - np.log(b + 1e-300))) * dx
def fit(direction, m0):
    def obj(v):
        q = np.exp(log_n(x, v[0], np.exp(v[1])))
        return kl(q, p) if direction == "reverse" else kl(p, q)
    v = minimize(obj, [m0, 0.0], method="Nelder-Mead").x
    return v[0], np.exp(v[1])
for d in ("reverse", "forward"):
    m, s = fit(d, 0.5)
    m = round(m, 2) + 0.0                       # -0.00 표기 방지
    print(f"KL({'q||p' if d == 'reverse' else 'p||q'}) 최소화: 평균 {m:5.2f}  sd {s:.2f}")
print(f"참 분포: 평균  0.00  sd {np.sqrt(np.sum(x**2 * p) * dx):.2f}")
