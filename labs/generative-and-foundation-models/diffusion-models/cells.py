"""19.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 6은 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/generative-and-foundation-models/diffusion-models/cells.py
"""
# %% cell 1 — forward 과정: 한 step씩 더한 잡음과 닫힌 형태의 주변분포
import numpy as np

T = 1000
betas = np.linspace(1e-4, 0.02, T)                # (T,) 선형 잡음 스케줄, t = 1..T
abar = np.cumprod(1 - betas)                       # (T,) abar_t = prod_{s<=t} (1 - beta_s)
PI, MU, SD = np.array([0.35, 0.65]), np.array([-1.5, 1.2]), np.array([0.25, 0.35])

def sample_x0(n, rng):                             # 데이터: 1차원 가우시안 혼합
    k = (rng.random(n) < PI[1]).astype(int)
    return MU[k] + SD[k] * rng.normal(size=n)

def comps(t):                                      # p_t도 혼합: 성분 평균·분산 (t = 0이면 데이터)
    a = 1.0 if t == 0 else abar[t - 1]
    return np.sqrt(a) * MU, a * SD**2 + (1 - a)

rng = np.random.default_rng(0)
x = sample_x0(20000, rng)                          # (20000,)
for t in range(1, T + 1):                          # q(x_t | x_{t-1}) = N(sqrt(1-beta_t) x_{t-1}, beta_t)
    x = np.sqrt(1 - betas[t - 1]) * x + np.sqrt(betas[t - 1]) * rng.normal(size=x.size)
    if t in (100, 300, 600, 1000):
        m, v = comps(t)
        mean = PI @ m
        var = PI @ (v + m**2) - mean**2
        print(f"t={t:4d}  abar={abar[t - 1]:.4f}  시뮬레이션 평균 {x.mean():6.3f} 분산 {x.var():.3f}"
              f"  |  닫힌 형태 평균 {mean:6.3f} 분산 {var:.3f}")

# %% cell 2 — denoising 목적함수의 최적해는 score다
def score(x, t):                                   # d/dx log p_t(x), 혼합의 닫힌 형태
    m, v = comps(t)
    logw = np.log(PI) - 0.5 * (x[:, None] - m) ** 2 / v - 0.5 * np.log(v)    # (n, 2)
    r = np.exp(logw - logw.max(1, keepdims=True))
    r /= r.sum(1, keepdims=True)                   # 성분 책임도
    return (r * (-(x[:, None] - m) / v)).sum(1)

def eps_star(x, t):                                # E[eps | x_t] = -sqrt(1 - abar_t) * score
    return -np.sqrt(1 - abar[t - 1]) * score(x, t)

t = 300
x0 = sample_x0(200000, rng)
eps = rng.normal(size=x0.size)
xt = np.sqrt(abar[t - 1]) * x0 + np.sqrt(1 - abar[t - 1]) * eps    # 닫힌 형태로 한 번에
for c in [0.0, 0.8, 1.0, 1.2]:
    print(f"eps_hat = {c} * eps*:  denoising 손실 E(eps - eps_hat)^2 = {np.mean((eps - c * eps_star(xt, t)) ** 2):.4f}")
x0_hat = (xt - np.sqrt(1 - abar[t - 1]) * eps_star(xt, t)) / np.sqrt(abar[t - 1])   # Tweedie 공식
print(f"x0 예측의 제곱오차 {np.mean((x0 - x0_hat) ** 2):.4f}  vs  x0 = x_t / sqrt(abar) {np.mean((x0 - xt / np.sqrt(abar[t - 1])) ** 2):.4f}")

# %% cell 3 — DDPM ancestral sampling: 정확한 score로, step 수를 바꿔 가며
from scipy.stats import wasserstein_distance

ref = sample_x0(200000, np.random.default_rng(1))  # 비교용 참 분포 표본

def sampler(n_steps, eta, n=10000, seed=2, ab_sampler=None):
    """eta = 1: DDPM(확률적), eta = 0: DDIM(결정론적). T개 시각 중 n_steps개만 쓴다."""
    ab_s = abar if ab_sampler is None else ab_sampler   # 샘플러가 믿는 abar
    g = np.random.default_rng(seed)
    ts = np.linspace(T, 1, n_steps).round().astype(int)
    x = g.normal(size=n)                           # x_T ~ N(0, 1)
    for i, t in enumerate(ts):
        a_t = ab_s[t - 1]
        a_prev = ab_s[ts[i + 1] - 1] if i + 1 < len(ts) else 1.0
        e = eps_star(x, t)                         # '학습된 신경망' 자리에 정확한 eps*
        x0_hat = (x - np.sqrt(1 - a_t) * e) / np.sqrt(a_t)
        sig = eta * np.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        x = np.sqrt(a_prev) * x0_hat + np.sqrt(1 - a_prev - sig**2) * e + sig * g.normal(size=n)
    return x

def report(name, s):
    print(f"{name:22s} W1 {wasserstein_distance(s, ref):.3f}  왼쪽 모드 비율 {np.mean(s < 0):.3f}  (참 0.350)")

for n_steps in [1000, 100, 20, 5]:
    report(f"DDPM {n_steps:4d} step", sampler(n_steps, eta=1.0))

# %% cell 4 — DDIM(결정론적)과 step 수, 그리고 잡음 스케줄 불일치
for n_steps in [100, 20, 5]:
    report(f"DDIM {n_steps:4d} step", sampler(n_steps, eta=0.0))
abar_b = np.cumprod(1 - np.linspace(1e-4, 0.01, T))     # 샘플러만 다른 스케줄(beta 끝값 절반)을 믿는다
report("DDPM 1000 step, 불일치", sampler(1000, eta=1.0, ab_sampler=abar_b))
report("DDIM   20 step, 불일치", sampler(20, eta=0.0, ab_sampler=abar_b))

# %% cell 5 — classifier-free guidance: 조건부 score와 무조건 score의 외삽
def eps_cond(x, t, k):                             # 조건 c = 성분 k일 때 p_t(x | c)는 가우시안
    m, v = comps(t)
    return -np.sqrt(1 - abar[t - 1]) * (-(x - m[k]) / v[k])

def guided_ddim(k, w, n_steps=50, n=10000, seed=3):
    g = np.random.default_rng(seed)
    ts = np.linspace(T, 1, n_steps).round().astype(int)
    x = g.normal(size=n)
    for i, t in enumerate(ts):
        e = (1 + w) * eps_cond(x, t, k) - w * eps_star(x, t)    # guidance 가중치 w
        a_t, a_prev = abar[t - 1], (abar[ts[i + 1] - 1] if i + 1 < len(ts) else 1.0)
        x0_hat = (x - np.sqrt(1 - a_t) * e) / np.sqrt(a_t)
        x = np.sqrt(a_prev) * x0_hat + np.sqrt(1 - a_prev) * e
    return x

print("조건 c = 오른쪽 성분, 참 조건부분포 N(1.2, 0.35^2)")
for w in [0.0, 1.0, 3.0]:
    s = guided_ddim(1, w)
    print(f"w = {w}: 평균 {s.mean():.3f}  표준편차 {s.std():.3f}  x < 0.7인 비율 {np.mean(s < 0.7):.3f}  (참 조건부분포 0.076)")

# %% cell 6 — 작은 신경망으로 eps를 학습해 DDPM으로 뽑는다
import numpy as np
import torch
from torch import nn
from scipy.stats import wasserstein_distance

torch.manual_seed(0)
torch.set_num_threads(1)                           # 작은 모델은 단일 스레드가 빠르다
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
abar = torch.cumprod(1 - betas, 0)                 # (T,)
def sample_x0(n):                                  # 0.35 N(-1.5, 0.25^2) + 0.65 N(1.2, 0.35^2)
    right = torch.rand(n) < 0.65
    return torch.where(right, 1.2 + 0.35 * torch.randn(n), -1.5 + 0.25 * torch.randn(n))
net = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
feat = lambda x, t: torch.stack([x, torch.sin(torch.pi * t / T), torch.cos(torch.pi * t / T)], 1)  # (B, 3)
opt = torch.optim.Adam(net.parameters(), 2e-3)
for step in range(2000):
    x0, t, e = sample_x0(512), torch.randint(1, T + 1, (512,)), torch.randn(512)
    xt = abar[t - 1].sqrt() * x0 + (1 - abar[t - 1]).sqrt() * e   # 닫힌 형태의 forward
    loss = ((net(feat(xt, t)).squeeze(1) - e) ** 2).mean()        # denoising 손실
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():                              # DDPM ancestral sampling, 1000 step
    x = torch.randn(10000)
    for t in range(T, 0, -1):
        e = net(feat(x, torch.full_like(x, t))).squeeze(1)
        x = (x - betas[t - 1] / (1 - abar[t - 1]).sqrt() * e) / (1 - betas[t - 1]).sqrt()
        if t > 1:                                  # 사후분포 분산 beta_t (1 - abar_{t-1}) / (1 - abar_t)
            x = x + (betas[t - 1] * (1 - abar[t - 2]) / (1 - abar[t - 1])).sqrt() * torch.randn_like(x)
print(f"마지막 batch의 denoising 손실 {loss.item():.3f}")
print(f"학습된 eps, DDPM 1000 step: W1 {wasserstein_distance(x.numpy(), sample_x0(200000).numpy()):.3f}"
      f"  왼쪽 모드 비율 {(x < 0).float().mean():.3f}  (참 0.350)")
