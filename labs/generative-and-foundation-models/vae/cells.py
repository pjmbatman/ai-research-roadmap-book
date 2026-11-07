"""19.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 4는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/generative-and-foundation-models/vae/cells.py
"""
# %% cell 1 — 선형 가우시안 VAE: ELBO의 두 항과 정확한 log p(x)
import numpy as np

w, b, sigma = np.array([1.5, -0.8]), np.array([0.2, 0.1]), 0.5   # decoder p(x|z) = N(w z + b, sigma^2 I)
x = np.array([1.4, -0.9])                         # (2,) 관측 하나, z는 1차원

def elbo_terms(x, mu_q, s_q):                     # encoder q(z|x) = N(mu_q, s_q^2)
    r = x - b - w * mu_q
    rec = -np.log(2 * np.pi * sigma**2) - (r @ r + (w @ w) * s_q**2) / (2 * sigma**2)
    kl = 0.5 * (mu_q**2 + s_q**2 - 1 - np.log(s_q**2))           # KL(q || N(0, 1))
    return rec, kl

C = np.outer(w, w) + sigma**2 * np.eye(2)          # p(x) = N(b, w w^T + sigma^2 I)
d = x - b
log_px = -0.5 * (d @ np.linalg.solve(C, d) + np.log(np.linalg.det(2 * np.pi * C)))
s_post2 = sigma**2 / (sigma**2 + w @ w)            # 정확한 사후분포 p(z|x)의 분산
m_post = (w @ d) / (sigma**2 + w @ w)              # 와 평균
for name, (mq, sq) in [("정확한 사후분포", (m_post, np.sqrt(s_post2))),
                       ("어긋난 encoder", (0.3, 0.6))]:
    rec, kl = elbo_terms(x, mq, sq)
    kl_post = 0.5 * (np.log(s_post2 / sq**2) + (sq**2 + (mq - m_post) ** 2) / s_post2 - 1)
    print(f"{name}: 재구성 {rec:.4f}  KL {kl:.4f}  ELBO {rec - kl:.4f}  "
          f"log p(x) - ELBO {log_px - (rec - kl):.4f}  KL(q||p(z|x)) {kl_post:.4f}")
print(f"log p(x) = {log_px:.4f}")

# %% cell 2 — 재매개변수화: 표본을 거쳐 encoder 파라미터로 gradient를 보낸다
rng = np.random.default_rng(0)
mq, sq, S = 0.3, 0.6, 1000                         # encoder 출력, 표본 수

def rec_grads(eps):                                # z = mq + sq * eps,  eps ~ N(0, 1)
    z = mq + sq * eps                              # (S,)
    r = x[None, :] - b - z[:, None] * w            # (S, 2)
    dlog_dz = (r @ w) / sigma**2                   # d log p(x|z) / dz
    return dlog_dz.mean(), (dlog_dz * eps).mean()  # dz/dmq = 1, dz/dsq = eps

g_mu, g_s = rec_grads(rng.normal(size=S))
r0 = x - b - w * mq
print(f"d 재구성/d mu_q : 재매개변수화 {g_mu:.4f}   정확한 값 {(r0 @ w) / sigma**2:.4f}")
print(f"d 재구성/d s_q  : 재매개변수화 {g_s:.4f}   정확한 값 {-(w @ w) * sq / sigma**2:.4f}")
reps = np.array([rec_grads(rng.normal(size=10)) for _ in range(2000)])   # 표본 10개짜리 추정 2000번
print(f"표본 10개 추정의 표준편차: mu_q {reps[:, 0].std():.4f}  s_q {reps[:, 1].std():.4f}")

# %% cell 3 — posterior collapse: 1차원 선형 VAE의 최적해를 beta와 decoder 잡음으로 찾는다
v = 1.0                                            # 데이터 분산, x ~ N(0, v)

def best_objective(beta, sig2):                    # encoder (a, s)는 닫힌 꼴로 최적화, w는 격자 탐색
    ws = np.linspace(0.0, 2.0, 20001)              # decoder 가중치 후보
    a = ws / (ws**2 + beta * sig2)                 # 최적 encoder 평균 계수: mu_q = a x
    s2 = beta * sig2 / (ws**2 + beta * sig2)       # 최적 encoder 분산
    rec = -0.5 * np.log(2 * np.pi * sig2) - (v * (1 - ws * a) ** 2 + ws**2 * s2) / (2 * sig2)
    kl = 0.5 * (a**2 * v + s2 - 1 - np.log(s2))
    i = np.argmax(rec - beta * kl)
    return ws[i], kl[i]

for beta, sig2 in [(1, 0.1), (1, 0.5), (1, 1.2), (4, 0.1), (12, 0.1)]:
    w_opt, kl_opt = best_objective(beta, sig2)
    print(f"beta={beta:2d} sigma^2={sig2:3.1f}  beta*sigma^2={beta * sig2:4.1f}  "
          f"최적 |w| {w_opt:.3f}  KL {kl_opt:.3f}  sqrt(v - beta*sigma^2) {np.sqrt(max(v - beta * sig2, 0)):.3f}")

# %% cell 4 — 작은 VAE를 PyTorch로 학습: decoder 잡음에 따른 KL·재구성·표본
import torch
from torch import nn

torch.manual_seed(0)
torch.set_num_threads(1)                           # 작은 모델은 단일 스레드가 빠르다
ang = torch.arange(8) * 2 * torch.pi / 8
centers = 2.0 * torch.stack([ang.cos(), ang.sin()], 1)            # (8, 2) 원 위의 모드
data = centers[torch.randint(8, (4000,))] + 0.15 * torch.randn(4000, 2)

def train_vae(sig_dec, steps=800):                 # beta = 1: 고정 분산 가우시안 decoder의 ELBO
    enc = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 4))   # -> (mu, log s) 2차원씩
    dec = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 2))
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=1e-2)
    for _ in range(steps):
        xb = data[torch.randint(4000, (256,))]                    # (256, 2)
        mu, log_s = enc(xb).chunk(2, dim=1)                       # (256, 2) 각각
        z = mu + log_s.exp() * torch.randn_like(mu)               # 재매개변수화
        rec = -((xb - dec(z)) ** 2).sum(1) / (2 * sig_dec**2)     # 상수항 생략
        kl = 0.5 * (mu**2 + (2 * log_s).exp() - 1 - 2 * log_s).sum(1)
        opt.zero_grad(); (kl - rec).mean().backward(); opt.step()
    with torch.no_grad():
        mu, log_s = enc(data).chunk(2, dim=1)
        kl_dim = 0.5 * (mu**2 + (2 * log_s).exp() - 1 - 2 * log_s).mean(0)   # 잠재 차원별 KL
        rmse = ((data - dec(mu)) ** 2).sum(1).mean().sqrt()       # encoder 평균으로 재구성
        xm = dec(torch.randn(2000, 2))                            # prior 표본의 decoder 평균
        near = (torch.cdist(xm, centers).min(1).values < 0.3).float().mean()
    print(f"sigma_dec={sig_dec:<4} 차원별 KL {kl_dim.numpy().round(2)}  재구성 RMSE {rmse:.2f}  "
          f"decoder 평균: 반지름 {xm.norm(dim=1).mean():.2f}, 모드 0.3 이내 {near:.2f}")

for sig_dec in [0.1, 0.5, 2.0]:
    train_vae(sig_dec)
