"""19.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 6은 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/generative-and-foundation-models/flow-matching/cells.py
"""
# %% cell 1 — 속도장으로 가우시안을 옮기고 밀도 변화를 발산으로 추적
import numpy as np

rng = np.random.default_rng(0)
mu, s = 2.0, 0.5                                   # 목표 분포 N(2, 0.5²)

def v(x, t):                                       # N(0,1)을 N(mu, s²)로 옮기는 속도장
    return mu + (s - 1) * (x - t * mu) / (1 - t + t * s)

x = rng.normal(size=5000)                          # x_0 ~ N(0, 1), (N,)
logp = -0.5 * x**2 - 0.5 * np.log(2 * np.pi)       # log p_0(x_0), (N,)
n_steps = 100
dt = 1.0 / n_steps
for k in range(n_steps):
    t = k * dt
    logp -= (s - 1) / (1 - t + t * s) * dt         # d log p / dt = -∂v/∂x
    x = x + v(x, t) * dt                           # Euler step
exact = -0.5 * ((x - mu) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)
print(f"x_1 평균 {x.mean():.3f}, 표준편차 {x.std():.3f}  (목표 {mu}, {s})")
print(f"log p_1 오차 최대 {np.abs(logp - exact).max():.4f}  (Euler {n_steps} step)")

# %% cell 2 — 조건부 flow matching: 속도장 학습은 최소제곱 회귀다
from scipy.stats import wasserstein_distance

rng = np.random.default_rng(1)
pis, ms, ss = np.array([0.3, 0.7]), np.array([-2.0, 2.0]), np.array([0.3, 0.3])   # 데이터: 혼합 가우시안
def sample_data(n):
    k = rng.choice(2, size=n, p=pis)
    return ms[k] + ss[k] * rng.normal(size=n)

def v_star(x, t):                                  # 참 주변 속도장 E[x1 - x0 | x_t = x] (닫힌 형태)
    var = (1 - t) ** 2 + t**2 * ss[:, None] ** 2   # 성분별 x_t의 분산 (2, N)
    logw = np.log(pis)[:, None] - 0.5 * (x - t * ms[:, None]) ** 2 / var - 0.5 * np.log(var)
    w = np.exp(logw - logw.max(0)); w /= w.sum(0)  # 사후 성분 확률 (2, N)
    return (w * (ms[:, None] + (t * ss[:, None] ** 2 - (1 - t)) / var * (x - t * ms[:, None]))).sum(0)

W_rf, b_rf = rng.normal(size=(2, 200)) * [[1.5], [3.0]], rng.uniform(0, 2 * np.pi, 200)
feat = lambda x, t: np.cos(np.stack([x, t * np.ones_like(x)], 1) @ W_rf + b_rf)   # (N, 200) 무작위 특징

def cfm_data(n, x1=None, x0=None):                 # x_t = (1-t)x0 + t x1, 목표 u = x1 - x0
    x0 = rng.normal(size=n) if x0 is None else x0
    x1 = sample_data(n) if x1 is None else x1
    t = rng.uniform(size=n)
    return (1 - t) * x0 + t * x1, t, x1 - x0

def fit(xt, t, u, lam=1e-3):                       # CFM 손실의 최소화 = ridge 회귀의 해
    F = feat(xt, t)
    return np.linalg.solve(F.T @ F + lam * len(u) * np.eye(F.shape[1]), F.T @ u)

theta = fit(*cfm_data(20000))
v_theta = lambda x, t: feat(x, t) @ theta
xt_te, t_te, u_te = cfm_data(20000)                # test용 (x_t, t, u)
loss = lambda f: np.mean((f(xt_te, t_te) - u_te) ** 2)
print(f"CFM 손실 (test)   v = 0: {loss(lambda x, t: 0 * x):.3f}   학습한 v_θ: {loss(v_theta):.3f}   참 v*: {loss(v_star):.3f}")
print(f"E|v_θ - v*|² = {np.mean((v_theta(xt_te, t_te) - v_star(xt_te, t_te)) ** 2):.3f}")

# %% cell 3 — step 수: solver 오차와 모델 오차를 나눈다
def euler(vf, x0, n_steps):                        # dx/dt = v(x, t), t: 0 → 1
    x, dt = x0.copy(), 1.0 / n_steps
    for k in range(n_steps):
        x = x + vf(x, np.full_like(x, k * dt)) * dt
    return x

x0 = rng.normal(size=2000)                         # 모든 설정에 같은 x0 (2000,)
ref = euler(v_star, x0, 1000)                      # 기준: 참 속도장을 아주 잘게 푼 결과
print("step   W1(참 v*)   W1(학습한 v_θ)")
for n in [1, 2, 4, 8, 32, 128]:
    print(f"{n:4d}     {wasserstein_distance(euler(v_star, x0, n), ref):.3f}       {wasserstein_distance(euler(v_theta, x0, n), ref):.3f}")

# %% cell 4 — reflow: 학습한 흐름이 정한 짝으로 다시 학습한다
z0 = rng.normal(size=12000)                        # 새 잡음 (12000,)
z1 = euler(v_theta, z0, 32)                        # 학습한 흐름이 정한 짝 z0 → z1 (결정적 coupling)
tr, te = slice(0, 8000), slice(8000, None)
theta_r = fit(*cfm_data(8000, x1=z1[tr], x0=z0[tr]))   # 같은 직선 경로, 짝만 바꾼다
v_reflow = lambda x, t: feat(x, t) @ theta_r
xt_q, t_q, u_q = cfm_data(4000, x1=z1[te], x0=z0[te])
print(f"CFM 손실: 독립 짝 {loss(v_theta):.3f}  →  reflow 짝 {np.mean((v_reflow(xt_q, t_q) - u_q) ** 2):.3f}")
print("step   W1(1차 흐름 v_θ)   W1(reflow)")
for n in [1, 2, 4, 32]:
    print(f"{n:4d}       {wasserstein_distance(euler(v_theta, x0, n), ref):.3f}            {wasserstein_distance(euler(v_reflow, x0, n), ref):.3f}")

# %% cell 5 — 확산 모델과의 관계: score·ε-예측·속도는 서로 바뀐다
def score(x, t):                                   # ∇x log p_t(x), 혼합 가우시안의 닫힌 형태
    var = (1 - t) ** 2 + t**2 * ss[:, None] ** 2
    logw = np.log(pis)[:, None] - 0.5 * (x - t * ms[:, None]) ** 2 / var - 0.5 * np.log(var)
    w = np.exp(logw - logw.max(0)); w /= w.sum(0)
    return (w * (-(x - t * ms[:, None]) / var)).sum(0)

xg = np.linspace(-4, 4, 201)                       # 격자 (201,)
for t in [0.1, 0.5, 0.9]:
    a, sig, da, dsig = t, 1 - t, 1.0, -1.0         # x_t = a·x1 + sig·ε 와 시간 미분
    sc = score(xg, t)
    v_from_score = da / a * xg + (da / a * sig**2 - dsig * sig) * sc
    eps_hat = -sig * sc                            # 확산 모델의 ε-예측 E[ε | x_t]
    x1_hat = (xg - sig * eps_hat) / a              # denoiser E[x1 | x_t]
    ok1 = np.abs(v_from_score - v_star(xg, t)).max() < 1e-8
    ok2 = np.abs((x1_hat - xg) / (1 - t) - v_star(xg, t)).max() < 1e-8
    print(f"t = {t}:  score로 만든 v = v*: {ok1}   (x̂1 - x)/(1-t) = v*: {ok2}")

# %% cell 6 — PyTorch: 2차원 데이터에서 경로 두 가지를 학습하고 step 수별 품질 비교
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(0)
rng = np.random.default_rng(0)
ang = 2 * np.pi * np.arange(8) / 8
centers = 2.0 * np.stack([np.cos(ang), np.sin(ang)], 1)           # 원 위의 가우시안 8개 (8, 2)
data = lambda n: torch.tensor(centers[rng.integers(8, size=n)] + 0.15 * rng.normal(size=(n, 2)), dtype=torch.float32)
half_pi = np.pi / 2
paths = {                                                          # x_t = a(t)·x1 + s(t)·ε 와 시간 미분
    "직선": (lambda t: t, lambda t: 1 - t, lambda t: 1 + 0 * t, lambda t: -1 + 0 * t),
    "삼각(VP류)": (lambda t: torch.sin(half_pi * t), lambda t: torch.cos(half_pi * t),
                 lambda t: half_pi * torch.cos(half_pi * t), lambda t: -half_pi * torch.sin(half_pi * t)),
}
proj = torch.nn.functional.normalize(torch.randn(2, 64), dim=0)    # 무작위 방향 64개 (2, 64)
sliced_w1 = lambda a, b: (torch.sort(a @ proj, 0)[0] - torch.sort(b @ proj, 0)[0]).abs().mean().item()
ref = data(4000)
print(f"기준: 데이터 두 표본의 sliced W1 = {sliced_w1(data(4000), ref):.3f}")
print("경로        끝 손실   1 step  2 step  4 step  8 step  32 step")
for name, (a, s, da, ds) in paths.items():
    net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    opt, tail = torch.optim.Adam(net.parameters(), lr=2e-3), []
    for step in range(3000):
        x1, eps, t = data(256), torch.randn(256, 2), torch.rand(256, 1)    # (B, 2), (B, 2), (B, 1)
        xt, u = a(t) * x1 + s(t) * eps, da(t) * x1 + ds(t) * eps          # 조건부 경로와 목표 속도
        loss = ((net(torch.cat([xt, t], 1)) - u) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tail.append(loss.item())
    w1 = []
    with torch.no_grad():
        for n in [1, 2, 4, 8, 32]:
            x = torch.randn(4000, 2, generator=torch.Generator().manual_seed(1))
            for k in range(n):
                x = x + net(torch.cat([x, torch.full((4000, 1), k / n)], 1)) / n
            w1.append(sliced_w1(x, ref))
    print(f"{name:10} {np.mean(tail[-300:]):7.3f}  " + "  ".join(f"{w:6.3f}" for w in w1))
