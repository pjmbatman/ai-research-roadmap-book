"""19.4 실습: 같은 2차원 데이터에 GAN과 affine coupling flow(RealNVP 형)를 학습해 비교한다.

데이터: 원 위의 모드 8개(반지름 2, 모드 표준편차 0.15). 19.1장과 같은 분포다.
비교: 모드 0.3 이내 표본 비율, 덮은 모드 수, test NLL(flow는 변수변환 공식으로 정확히, GAN은 계산 불가),
      그리고 두 모델의 표본에 같은 커널 밀도 추정(KDE)을 적용한 test NLL.

실행(저장소 루트에서, CPU 1분 안팎):
    uv run --project labs --group torch python labs/generative-and-foundation-models/gan-and-flows/gan_vs_flow.py
"""
import math

import numpy as np
import torch
from scipy.special import logsumexp
from torch import nn

torch.manual_seed(0)
torch.set_num_threads(1)
K, SD = 8, 0.15
ang = torch.arange(K) * 2 * torch.pi / K
CENTERS = 2.0 * torch.stack([ang.cos(), ang.sin()], 1)          # (8, 2)


def data(n):
    return CENTERS[torch.randint(K, (n,))] + SD * torch.randn(n, 2)


def true_nll(x):                                                 # 참 분포의 NLL (기준값)
    d2 = ((x[:, None, :] - CENTERS[None]) ** 2).sum(-1)
    lp = torch.logsumexp(-d2 / (2 * SD**2) - math.log(2 * math.pi * SD**2), 1) - math.log(K)
    return -lp.mean().item()


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


# ---------------- GAN ----------------
def train_gan(steps=3000):
    G, D = mlp(2, 2), mlp(2, 1)
    og = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), 1e-3, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    ones, zeros = torch.ones(256, 1), torch.zeros(256, 1)
    for _ in range(steps):
        xf = G(torch.randn(256, 2))
        ld = bce(D(data(256)), ones) + bce(D(xf.detach()), zeros)
        od.zero_grad(); ld.backward(); od.step()
        lg = bce(D(xf), ones)
        og.zero_grad(); lg.backward(); og.step()
    return lambda n: G(torch.randn(n, 2))


# ---------------- affine coupling flow ----------------
class Coupling(nn.Module):
    """한 좌표(a)를 그대로 두고 다른 좌표(b)를 b * exp(s(a)) + t(a)로 바꾼다."""

    def __init__(self, flip):
        super().__init__()
        self.flip = flip
        self.net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))

    def _split(self, v):
        return (v[:, 1:], v[:, :1]) if self.flip else (v[:, :1], v[:, 1:])

    def _join(self, a, b):
        return torch.cat([b, a], 1) if self.flip else torch.cat([a, b], 1)

    def st(self, a):
        s, t = self.net(a).chunk(2, 1)
        return 2 * torch.tanh(s / 2), t                          # 스케일을 제한해 학습을 안정화

    def forward(self, z):                                        # z -> x
        a, b = self._split(z)
        s, t = self.st(a)
        return self._join(a, b * s.exp() + t)

    def inverse(self, x):                                        # x -> z, log|det dz/dx|
        a, b = self._split(x)
        s, t = self.st(a)
        return self._join(a, (b - t) * (-s).exp()), -s.sum(1)


class Flow(nn.Module):
    def __init__(self, n_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([Coupling(i % 2 == 1) for i in range(n_layers)])

    def log_prob(self, x):                                       # 변수변환 공식
        logdet = torch.zeros(len(x))
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            logdet = logdet + ld
        return -0.5 * (x**2).sum(1) - math.log(2 * math.pi) + logdet

    def sample(self, n):
        z = torch.randn(n, 2)
        for layer in self.layers:
            z = layer(z)
        return z


def train_flow(steps=4000):
    flow = Flow()
    opt = torch.optim.Adam(flow.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        loss = -flow.log_prob(data(256)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return flow


# ---------------- 평가 ----------------
def sample_metrics(s):
    dist = torch.cdist(s, CENTERS)
    near = dist.min(1).values < 0.3
    frac = torch.bincount(dist.argmin(1), minlength=K).float() / len(s)   # 가장 가까운 모드별 비율
    modes = int((torch.bincount(dist.argmin(1)[near], minlength=K) > 0.01 * len(s)).sum())
    return near.float().mean().item(), modes, frac.min().item(), frac.max().item()


def kde_nll(samples, x, h=0.1):
    s, x = samples.numpy().astype(float), x.numpy().astype(float)
    d2 = ((x[:, None, :] - s[None]) ** 2).sum(-1)
    lp = logsumexp(-d2 / (2 * h**2) - np.log(2 * np.pi * h**2), axis=1) - np.log(len(s))
    return -lp.mean()


def main():
    x_test = data(2000)
    gan_sampler = train_gan()
    flow = train_flow()
    with torch.no_grad():
        rows = [("참 분포", data(4000), true_nll(x_test)),
                ("GAN", gan_sampler(4000), None),
                ("flow", flow.sample(4000), -flow.log_prob(x_test).mean().item())]
        print("모델    모드 0.3 이내  덮은 모드  모드별 비율(최소–최대)  표본 공백  test NLL(정확)  test NLL(KDE)")
        for name, s, nll in rows:
            near, modes, fmin, fmax = sample_metrics(s)
            hole = (torch.cdist(x_test, s).min(1).values > 0.1).float().mean().item()
            exact = f"{nll:.3f}" if nll is not None else "계산 불가"
            print(f"{name:6s}  {near:10.2f}  {modes:6d}/8  {fmin:10.3f}–{fmax:.3f}  {hole:9.3f}"
                  f"  {exact:>12s}  {kde_nll(s, x_test):10.3f}")
        print("(표본 공백: 0.1 이내에 표본이 하나도 없는 test 점의 비율. KDE: 표본 4000개, 커널 폭 0.1."
              " 참 분포의 모드별 비율은 0.125)")


if __name__ == "__main__":
    main()
