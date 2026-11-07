"""19.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 4는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/generative-and-foundation-models/gan-and-flows/cells.py
"""
# %% cell 1 — 최적 판별자와 Jensen–Shannon divergence, 그리고 gradient 소실
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

x = np.linspace(-12, 16, 7001)                     # (X,) 적분 격자
dx = x[1] - x[0]
log_pd = logsumexp([norm.logpdf(x, -2, 0.5), norm.logpdf(x, 2, 0.5)], axis=0) - np.log(2)
p_data = np.exp(log_pd)                            # 0.5 N(-2, 0.5^2) + 0.5 N(2, 0.5^2)

def value_at_optimal_d(log_pg):                    # V(D*, G) = E_data log D* + E_g log(1 - D*)
    log_den = np.logaddexp(log_pd, log_pg)         # D* = p_data / (p_data + p_g), 로그로 계산
    return np.sum(p_data * (log_pd - log_den) + np.exp(log_pg) * (log_pg - log_den)) * dx

def js(log_p, log_q):                              # 0.5 KL(p||m) + 0.5 KL(q||m), m = (p + q)/2
    log_m = np.logaddexp(log_p, log_q) - np.log(2)
    return 0.5 * np.sum(np.exp(log_p) * (log_p - log_m) + np.exp(log_q) * (log_q - log_m)) * dx

for mu in [0.0, 3.0, 4.5, 8.0]:
    v = value_at_optimal_d(norm.logpdf(x, mu, 0.5))
    grad = (value_at_optimal_d(norm.logpdf(x, mu + 0.01, 0.5))
            - value_at_optimal_d(norm.logpdf(x, mu - 0.01, 0.5))) / 0.02
    two_js = 2 * js(log_pd, norm.logpdf(x, mu, 0.5)) - np.log(4)
    print(f"생성자 N({mu:4.1f}, 0.5^2): V(D*,G) {round(v, 4) + 0.0:7.4f}  2JS - log4 {round(two_js, 4) + 0.0:7.4f}"
          f"  dV/dmu {round(grad, 3) + 0.0:6.3f}")

# %% cell 2 — 가우시안 하나로 쌍봉 분포를 맞출 때: divergence마다 다른 최적해
def kl(log_p, log_q):
    return np.sum(np.exp(log_p) * (log_p - log_q)) * dx

mus, sds = np.linspace(-3, 3, 61), np.linspace(0.2, 3.0, 29)
for name, div in [("KL(data||g) (최대우도)", lambda lq: kl(log_pd, lq)),
                  ("JS", lambda lq: js(log_pd, lq)),
                  ("KL(g||data)", lambda lq: kl(lq, log_pd))]:
    vals = np.array([[div(norm.logpdf(x, m, s)) for s in sds] for m in mus])   # (61, 29)
    i, j = np.unravel_index(np.argmin(vals), vals.shape)
    print(f"{name:22s} 최적 생성자: |평균| {abs(mus[i]):.2f}, 표준편차 {sds[j]:.2f}  divergence {vals[i, j]:.3f}")

# %% cell 3 — 작은 GAN의 학습 동역학: 한 모드에 갇힌 생성자
rng = np.random.default_rng(0)
C = np.linspace(-4, 4, 17)                         # 판별자 RBF 특징의 중심
phi = lambda v: np.exp(-(v[:, None] - C) ** 2 / (2 * 0.5**2))           # (n, 17)
sig = lambda z: 1 / (1 + np.exp(-z))
w, b, mu, s = np.zeros(17), 0.0, -0.3, 0.3        # 판별자 (w, b), 생성자 x = mu + s * eps
for t in range(601):
    xr = np.where(rng.random(128) < 0.5, -2, 2) + 0.5 * rng.normal(size=128)
    eps = rng.normal(size=128)
    xf = mu + s * eps                              # (128,) 생성 표본
    for _ in range(2):                             # 판별자: log D(real) + log(1 - D(fake)) 상승
        pr, pf = sig(phi(xr) @ w + b), sig(phi(xf) @ w + b)
        w += phi(xr).T @ (1 - pr) / 128 - phi(xf).T @ pf / 128
        b += np.mean(1 - pr) - np.mean(pf)
    pf = sig(phi(xf) @ w + b)                      # 생성자: log D(fake) 상승 (non-saturating)
    dlogit = (phi(xf) * (-(xf[:, None] - C) / 0.25)) @ w                  # d logit / d x
    mu += 0.05 * np.mean((1 - pf) * dlogit)
    s *= np.exp(0.05 * np.mean((1 - pf) * dlogit * eps) * s)              # log s 갱신
    if t % 150 == 0:
        print(f"step {t:3d}: 생성자 N({mu:5.2f}, {s:.2f}^2)")
print("판별자 D(x) at x = -2, 0, 2:", np.round(sig(phi(np.array([-2.0, 0.0, 2.0])) @ w + b), 3))
print(f"JS: 학습 결과 {js(log_pd, norm.logpdf(x, mu, s)):.3f}  vs  JS 최적 가우시안 {js(log_pd, norm.logpdf(x, 0, 2.0)):.3f}")

# %% cell 4 — PyTorch GAN: 손실 값과 표본 품질은 따로 움직인다
import torch
from torch import nn

torch.manual_seed(0)
torch.set_num_threads(1)                           # 작은 모델은 단일 스레드가 빠르다
ang = torch.arange(8) * 2 * torch.pi / 8
centers = 2.0 * torch.stack([ang.cos(), ang.sin()], 1)                  # (8, 2) 원 위의 모드
data = lambda n: centers[torch.randint(8, (n,))] + 0.15 * torch.randn(n, 2)
mlp = lambda out: nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, out))
G, D = mlp(2), mlp(1)                              # 생성자 z (2,) -> x (2,), 판별자 x -> logit
opt_g = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), 1e-3, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
ones, zeros = torch.ones(256, 1), torch.zeros(256, 1)
for step in range(1, 2001):
    xf = G(torch.randn(256, 2))                    # (256, 2)
    loss_d = bce(D(data(256)), ones) + bce(D(xf.detach()), zeros)
    opt_d.zero_grad(); loss_d.backward(); opt_d.step()
    loss_g = bce(D(xf), ones)                      # non-saturating 생성자 손실
    opt_g.zero_grad(); loss_g.backward(); opt_g.step()
    if step % 400 == 0:
        with torch.no_grad():
            dist = torch.cdist(G(torch.randn(2000, 2)), centers)          # (2000, 8)
            near = dist.min(1).values < 0.3
            modes = (torch.bincount(dist.argmin(1)[near], minlength=8) > 20).sum()
        print(f"step {step:4d}  D 손실 {loss_d.item():.3f}  G 손실 {loss_g.item():.3f}"
              f"  모드 0.3 이내 {near.float().mean():.2f}  덮은 모드 {modes}/8")

# %% cell 5 — affine coupling 두 층: 역변환, log-det, 전체 질량
import numpy as np

def s_net(u): return 0.6 * np.tanh(u)              # 고정된 '신경망' (학습 없이 구조만 확인)
def t_net(u): return 1.2 * np.sin(1.5 * u)

def forward(z):                                    # z (n, 2) -> x (n, 2), log|det dx/dz|
    x1, x2 = z[:, 0], z[:, 1] * np.exp(s_net(z[:, 0])) + t_net(z[:, 0])   # 층 1: x1 고정
    y2, y1 = x2, x1 * np.exp(s_net(x2)) + t_net(x2)                      # 층 2: x2 고정
    return np.stack([y1, y2], 1), s_net(z[:, 0]) + s_net(x2)

def inverse(y):                                    # 층 순서를 거꾸로, 각 층은 닫힌 꼴
    x2 = y[:, 1]
    x1 = (y[:, 0] - t_net(x2)) * np.exp(-s_net(x2))
    z1 = x1
    return np.stack([z1, (x2 - t_net(z1)) * np.exp(-s_net(z1))], 1)

def log_prob(y):                                   # log p(y) = log N(z; 0, I) - log|det dy/dz|
    z = inverse(y)
    return -0.5 * (z**2).sum(1) - np.log(2 * np.pi) - forward(z)[1]

z = np.random.default_rng(1).normal(size=(5, 2))
y, logdet = forward(z)
J = np.stack([(forward(z + e)[0] - forward(z - e)[0]) / 2e-6 for e in np.eye(2) * 1e-6], 2)  # (5, 2, 2)
print("역변환 오차 < 1e-12:", np.abs(inverse(y) - z).max() < 1e-12)
print("log|det J| 공식 - 수치 Jacobian, 최대 차이 < 1e-6:", np.abs(logdet - np.log(np.abs(np.linalg.det(J)))).max() < 1e-6)
g = np.linspace(-10, 10, 1001)
Y = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)  # (1001^2, 2) 격자
print(f"격자로 적분한 전체 질량: {np.exp(log_prob(Y)).sum() * (g[1] - g[0]) ** 2:.4f}")
