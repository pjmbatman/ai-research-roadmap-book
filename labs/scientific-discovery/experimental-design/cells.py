"""25.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/scientific-discovery/experimental-design/cells.py
"""
# %% cell 1 — 능동 학습: 불확실성 샘플링 대 무작위 선택
import numpy as np

def sim(x):                                           # "비싼 시뮬레이터" 대역: 응답 곡선
    return np.sin(6 * x) + 0.5 * np.sin(17 * x) * np.exp(-3 * x)
grid = np.linspace(0, 1, 201)                         # (G,) 후보 실험 조건이자 평가 격자
NOISE = 0.05                                          # 측정 잡음 표준편차

def gp(X, y, Xs, ell, noise=NOISE):                   # RBF 커널 GP의 사후 평균·표준편차
    k = lambda a, b: np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)
    L = np.linalg.cholesky(k(X, X) + noise**2 * np.eye(len(X)))   # (n, n)
    v = np.linalg.solve(L, k(X, Xs))                  # (n, G)
    mu = v.T @ np.linalg.solve(L, y)                  # (G,)
    return mu, np.sqrt(np.clip(1 - (v**2).sum(0), 1e-12, None))

def run(strategy, seed, budget=20, ell=0.08):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, 3)                          # 초기 실험 3개는 무작위
    y = sim(X) + NOISE * rng.normal(size=3)
    rmse = {}
    while True:
        mu, sd = gp(X, y, grid, ell)
        rmse[len(X)] = np.sqrt(np.mean((mu - sim(grid)) ** 2))   # 실험 n개일 때의 RMSE
        if len(X) == budget:
            return rmse
        x = grid[np.argmax(sd)] if strategy == "uncertainty" else rng.uniform(0, 1)
        X, y = np.append(X, x), np.append(y, sim(x) + NOISE * rng.normal())

for name, kw in [("무작위", dict(strategy="random")), ("불확실성", dict(strategy="uncertainty")),
                 ("불확실성(ell=0.3)", dict(strategy="uncertainty", ell=0.3))]:
    runs = [run(seed=s, **kw) for s in range(20)]                  # 시드 20개
    print(f"{name:12s} RMSE 중앙값  " + "  ".join(f"실험 {n}개 {np.median([r[n] for r in runs]):.3f}" for n in (6, 10, 20)))

# %% cell 2 — 선형 가우시안 모델의 기대 정보 이득(해석해)
SIGMA = 0.5                                           # 측정 잡음 표준편차
S0 = np.eye(2)                                        # 사전 공분산: theta = (절편, 기울기) ~ N(0, I)

def phi(x):                                           # (n,) -> (n, 2) 설계 행렬: y = theta0 + theta1 x
    return np.stack([np.ones_like(x), x], axis=1)

def eig_linear(x):                                    # EIG = 1/2 log det(I + Φ S0 Φ^T / σ^2) [nat]
    P = phi(x)
    return 0.5 * np.linalg.slogdet(np.eye(len(x)) + P @ S0 @ P.T / SIGMA**2)[1]

def post_cov(x):                                      # 사후 공분산 (S0^-1 + Φ^T Φ / σ^2)^-1
    P = phi(x)
    return np.linalg.inv(np.linalg.inv(S0) + P.T @ P / SIGMA**2)

designs = {"한곳에 4번 (0)": [0, 0, 0, 0], "고르게 4점": [-1, -1/3, 1/3, 1],
           "양 끝에 2번씩": [-1, -1, 1, 1]}
for name, x in designs.items():
    x = np.array(x, dtype=float)
    sd = np.sqrt(np.diag(post_cov(x)))
    print(f"{name:14s} EIG {eig_linear(x):.3f} nat   사후 표준편차 절편 {sd[0]:.3f}  기울기 {sd[1]:.3f}")

# %% cell 3 — 중첩 몬테카를로 EIG: 해석해 확인과 비선형 모델
from scipy.special import logsumexp

def eig_nmc(simulate_mean, sample_prior, sigma, N=1000, M=1000, seed=0):
    """EIG ≈ 평균_n [log p(y_n|θ_n) - log 평균_m p(y_n|θ_m)], 가우시안 측정 잡음."""
    rng = np.random.default_rng(seed)
    th_n, th_m = sample_prior(rng, N), sample_prior(rng, M)
    mu_n, mu_m = simulate_mean(th_n), simulate_mean(th_m)          # (N, n), (M, n)
    y = mu_n + sigma * rng.normal(size=mu_n.shape)                 # (N, n) 가상 측정
    outer = -0.5 * (((y - mu_n) / sigma) ** 2).sum(-1)             # (N,) log p(y_n|θ_n) + 상수
    cross = -0.5 * (((y[:, None, :] - mu_m[None]) / sigma) ** 2).sum(-1)   # (N, M) 같은 상수 생략
    inner = logsumexp(cross, axis=1) - np.log(M)                   # (N,) log 평균_m p(y_n|θ_m)
    return np.mean(outer - inner)

prior2 = lambda rng, k: rng.normal(size=(k, 2))                    # theta ~ N(0, I)
for name, x in designs.items():
    x = np.array(x, dtype=float)
    print(f"{name:14s} 해석해 {eig_linear(x):.3f}   중첩 MC {eig_nmc(lambda th: th @ phi(x).T, prior2, SIGMA):.3f}")

# 비선형: 감쇠 y = exp(-k t) + 잡음, 감쇠율 k ~ LogNormal(log 1, 0.5) [1/s]. 측정 시각 t_s 하나를 고른다
prior_k = lambda rng, n: np.exp(rng.normal(0.0, 0.5, size=(n, 1)))
for t_s in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    e = eig_nmc(lambda k: np.exp(-k * t_s), prior_k, sigma=0.05)
    print(f"측정 시각 {t_s:4.2f} s  EIG {e:.3f} nat")

# %% cell 4 — D-최적 설계: 교환 알고리즘
def logdet_info(x):                                   # 2차 회귀 y = b0 + b1 x + b2 x^2의 log det(X^T X)
    X = np.stack([np.ones_like(x), x, x**2], axis=1)  # (n, 3)
    return np.linalg.slogdet(X.T @ X)[1]

cand = np.linspace(-1, 1, 41)                         # 후보 실험 조건
rng = np.random.default_rng(0)
x = rng.choice(cand, size=6)                          # 무작위 초기 설계, 실험 6번
for sweep in range(10):                               # 한 점씩 다른 후보로 바꿔 보며 개선되면 교환
    improved = False
    for i in range(len(x)):
        scores = [logdet_info(np.r_[x[:i], c, x[i + 1:]]) for c in cand]
        j = int(np.argmax(scores))
        if scores[j] > logdet_info(x) + 1e-9:
            x[i], improved = cand[j], True
    if not improved:
        break
print("교환 알고리즘 설계:", np.sort(x) + 0.0, f" log det = {logdet_info(x):.3f}")
for name, xd in [("고르게 6점", np.linspace(-1, 1, 6)), ("가운데 몰림", np.linspace(-0.3, 0.3, 6))]:
    print(f"{name:10s}: log det = {logdet_info(xd):.3f}")

# %% cell 5 — 같은 GP, 다른 목표: 능동 학습 대 베이지안 최적화
from scipy.stats import norm

def run2(acq, seed, budget=12, ell=0.08):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, 3)
    y = sim(X) + NOISE * rng.normal(size=3)
    while len(X) < budget:
        mu, sd = gp(X, y, grid, ell)
        if acq == "불확실성":
            score = sd                                         # 모델이 가장 모르는 곳
        else:                                                  # EI: 최솟값을 낮출 기대량
            z = (mu.min() - mu) / sd
            score = (mu.min() - mu) * norm.cdf(z) + sd * norm.pdf(z)
        x = grid[np.argmax(score)]
        X, y = np.append(X, x), np.append(y, sim(x) + NOISE * rng.normal())
    mu, _ = gp(X, y, grid, ell)
    rmse = np.sqrt(np.mean((mu - sim(grid)) ** 2))             # 함수 전체를 얼마나 아는가
    regret = sim(X).min() - sim(grid).min()                    # 최솟값을 얼마나 찾았는가
    return rmse, regret

for acq in ["불확실성", "EI"]:
    R = np.array([run2(acq, s) for s in range(20)])            # (시드, 2)
    print(f"{acq:6s} 실험 12개 후  전체 RMSE 중앙값 {np.median(R[:, 0]):.3f}   regret 중앙값 {np.median(R[:, 1]):.3f}")
