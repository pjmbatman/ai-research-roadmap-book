"""21.1 실습: 딥 앙상블과 MC dropout의 불확실성을 입력 영역별로 비교한다.

실행: uv run --project labs --group torch python labs/reliability/uncertainty-quantification/compare_uq.py
CPU에서 1분 안쪽으로 끝난다.

데이터: y = sin(x) + ε, ε ~ N(0, sd(x)²).
  밀집·저잡음  x ∈ [-3, -1]  400개, sd 0.05
  밀집·고잡음  x ∈ [ 1,  3]  400개, sd 0.40
  희소        x ∈ [ 3,  4]   15개, sd 0.05
  빈 구간     x ∈ [-1,  1]    0개
  도메인 밖   x ∈ [ 5,  6]    0개
두 방법 모두 (μ, log σ²)를 출력하는 MLP를 이분산 가우시안 NLL로 학습한다.
"""
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)     # 작은 모델: 스레드 하나가 더 빠르고 결과도 재현된다
torch.manual_seed(0)
rng = np.random.default_rng(0)


def noise_sd(x):
    return np.where((x > 1) & (x < 3), 0.40, 0.05)


xs = np.concatenate([rng.uniform(-3, -1, 400), rng.uniform(1, 3, 400), rng.uniform(3, 4, 15)])
ys = np.sin(xs) + rng.normal(size=xs.size) * noise_sd(xs)
X = torch.tensor(xs, dtype=torch.float32)[:, None]   # (N, 1)
Y = torch.tensor(ys, dtype=torch.float32)[:, None]   # (N, 1)

REGIONS = {"밀집·저잡음": (-3, -1), "밀집·고잡음": (1, 3), "희소": (3, 4), "빈 구간": (-1, 1), "도메인 밖": (5, 6)}


def make_net(p_drop=0.0):
    layers = []
    for d_in in (1, 64):
        layers += [nn.Linear(d_in, 64), nn.Tanh()]
        if p_drop:
            layers.append(nn.Dropout(p_drop))
    return nn.Sequential(*layers, nn.Linear(64, 2))   # 출력: (μ, log σ²)


def train(net, epochs=2000, warmup=500):
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for ep in range(epochs):
        out = net(X)
        mu, logvar = out[:, :1], out[:, 1:].clamp(-12, 5)
        if ep < warmup:                       # 평균을 먼저 MSE로 맞춘다(이분산 손실의 불안정성 완화)
            loss = ((Y - mu) ** 2).mean()
        else:
            loss = (0.5 * logvar + 0.5 * (Y - mu) ** 2 / logvar.exp()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def predict(net, x, T=1, mc=False):
    net.train(mc)                              # mc=True면 예측 때도 dropout을 켠다
    with torch.no_grad():
        outs = torch.stack([net(x) for _ in range(T)])   # (T, n, 2)
    return outs[..., 0].numpy(), outs[..., 1].clamp(-12, 5).exp().numpy()


ens = [train(make_net()) for _ in range(5)]
mcd = train(make_net(p_drop=0.1))

print("영역별 평균 sd (aleatoric / epistemic),  참 잡음 sd")
print(f"{'영역':12s} {'앙상블 5개':>16s} {'MC dropout T=50':>18s}   참값")
for name, (lo, hi) in REGIONS.items():
    grid = np.linspace(lo, hi, 52)[1:-1]                 # 영역 안쪽 점 50개 (경계 제외)
    xg = torch.tensor(grid, dtype=torch.float32)[:, None]
    mu_e, var_e = map(np.concatenate, zip(*[predict(n, xg) for n in ens]))   # (5, 50)
    mu_d, var_d = predict(mcd, xg, T=50, mc=True)                            # (50, 50)
    a_e, e_e = np.sqrt(var_e.mean(0)).mean(), mu_e.std(0).mean()
    a_d, e_d = np.sqrt(var_d.mean(0)).mean(), mu_d.std(0).mean()
    true = f"{noise_sd(grid).mean():.2f}" if name not in ("빈 구간", "도메인 밖") else "(데이터 없음)"
    print(f"{name:12s} {a_e:7.3f} / {e_e:6.3f}   {a_d:7.3f} / {e_d:6.3f}   {true}")
