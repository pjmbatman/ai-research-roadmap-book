"""19.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/generative-and-foundation-models/generative-modeling/cells.py
"""
# %% cell 1 — 최대우도와 KL: 무조건 p(x)와 조건부 p(x|c)
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

rng = np.random.default_rng(0)
K, SD = 8, 0.15                                   # 원 위의 모드 8개, 모드 폭
ang = 2 * np.pi * np.arange(K) / K
CENTERS = 2.0 * np.stack([np.cos(ang), np.sin(ang)], axis=1)   # (K, 2)

def sample_data(n, rng):
    c = rng.integers(K, size=n)                   # (n,) 조건 c = 모드 번호
    return CENTERS[c] + SD * rng.normal(size=(n, 2)), c

def log_gauss(x, mu, sd):                         # 등방 가우시안 log N(x; mu, sd^2 I)
    return -((x - mu) ** 2).sum(-1) / (2 * sd**2) - np.log(2 * np.pi * sd**2)

x_tr, c_tr = sample_data(2000, rng)               # (2000, 2), (2000,)
x_te, c_te = sample_data(2000, rng)
mu_c = np.stack([x_tr[c_tr == k].mean(0) for k in range(K)])   # (K, 2) p(x|c)의 MLE
sd_c = np.sqrt(((x_tr - mu_c[c_tr]) ** 2).mean())              # 공유 표준편차의 MLE
log_px_c = log_gauss(x_te[:, None, :], mu_c[None], sd_c)       # (2000, K)
nll_cond = -log_px_c[np.arange(len(x_te)), c_te].mean()        # -log p(x|c)
nll_mix = -(logsumexp(log_px_c, axis=1) - np.log(K)).mean()    # p(x) = sum_c p(c) p(x|c)
mu, S = x_tr.mean(0), np.cov(x_tr.T)              # 가우시안 하나로 맞춘 p(x)
nll_gauss = -multivariate_normal(mu, S).logpdf(x_te).mean()
H = -(logsumexp(log_gauss(x_te[:, None, :], CENTERS[None], SD), axis=1) - np.log(K)).mean()
print(f"데이터 엔트로피 H(p_data) 추정   {H:.3f} nats")
print(f"가우시안 하나  NLL {nll_gauss:.3f}  KL 추정 {nll_gauss - H:.3f}")
print(f"8성분 혼합     NLL {nll_mix:.3f}  KL 추정 {nll_mix - H:.3f}")
print(f"조건부 p(x|c)  NLL {nll_cond:.3f}  (무조건 - 조건부 = {nll_mix - nll_cond:.3f}, log 8 = {np.log(8):.3f})")

# %% cell 2 — 샘플링만 되는 모델: 커널 밀도로 추정한 likelihood의 불안정성
from scipy.stats import norm

def implicit_sampler(n, rng):                     # x = g(z): 밀도 식 없이 표본만 만든다
    z = rng.normal(size=(n, 3))                   # (n, 3) 잠재 잡음
    k = np.minimum((norm.cdf(z[:, 0]) * K).astype(int), K - 1)
    return CENTERS[k] + SD * z[:, 1:]             # (n, 2)

def kde_loglik(x, samples, h):                    # 표본 위에 폭 h 가우시안을 얹은 밀도
    lp = log_gauss(x[:, None, :], samples[None], h)              # (n_test, n_samples)
    return (logsumexp(lp, axis=1) - np.log(len(samples))).mean()

print(f"참 평균 log-likelihood  {-H:.3f}")
for n in [200, 2000]:
    g = implicit_sampler(n, rng)
    row = "  ".join(f"h={h:<4} {kde_loglik(x_te, g, h):7.3f}" for h in [0.02, 0.08, 0.15, 0.4])
    print(f"표본 {n:5d}개  {row}")

# %% cell 3 — 생성과 예측: 조건부 평균은 표본이 아니다
band = np.abs(x_tr[:, 0]) < 0.3                   # x1이 0 근처인 데이터
y = x_tr[band, 1]                                 # (m,) 그때의 x2
pred = y.mean()                                   # 제곱오차를 최소화하는 점 예측
print(f"x1≈0인 데이터 {band.sum()}개, x2의 값: 위쪽 모드 {np.mean(y > 0):.2f}, 아래쪽 모드 {np.mean(y < 0):.2f}")
print(f"MSE 최적 예측 E[x2|x1≈0] = {pred:.3f}")
print(f"예측값에서 0.5 이내에 있는 실제 데이터 비율 = {np.mean(np.abs(y - pred) < 0.5):.3f}")
draw = rng.choice(y, size=6)                      # 조건부분포 p(x2|x1≈0)에서의 표본
print("조건부 표본 6개:", np.round(draw, 2))

# %% cell 4 — 같은 평가로 네 '모델' 비교: 보기 좋은 표본과 분포 적합은 다르다
def nearest_mode_dist(s):                         # 표본이 가장 가까운 모드에서 떨어진 거리
    return np.sqrt(((s[:, None, :] - CENTERS[None]) ** 2).sum(-1)).min(1)

def evaluate(name, samples, loglik_fn):
    mode = np.sqrt(((samples[:, None, :] - CENTERS[None]) ** 2).sum(-1)).argmin(1)
    cover = np.sum(np.bincount(mode, minlength=K) > 0.01 * len(samples))
    copy = np.mean(np.sqrt(((samples[:, None, :] - x_tr[None]) ** 2).sum(-1)).min(1) < 1e-3)
    print(f"{name:14s} 모드거리 {nearest_mode_dist(samples).mean():.3f}  모드 {cover}/8"
          f"  복제율 {copy:.2f}  test NLL {-loglik_fn(x_te):7.3f}")

def mix_loglik(centers):
    return lambda x: (logsumexp(log_gauss(x[:, None, :], centers[None], SD), axis=1)
                      - np.log(len(centers))).mean()

n = 1000
evaluate("참 분포", sample_data(n, rng)[0], mix_loglik(CENTERS))
evaluate("가우시안 하나", mu + rng.normal(size=(n, 2)) @ np.linalg.cholesky(S).T, lambda x: -nll_gauss)
evaluate("학습셋 복사", x_tr[rng.choice(len(x_tr), n)], lambda x: kde_loglik(x, x_tr, 0.01))
half = CENTERS[::2]                               # 모드 4개만 내는 모델
evaluate("모드 절반", half[rng.integers(4, size=n)] + SD * rng.normal(size=(n, 2)), mix_loglik(half))
