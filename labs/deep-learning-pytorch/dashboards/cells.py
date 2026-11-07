"""17.8 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 1은 PyTorch가 필요하다(exec="python"). cell 2–4는 NumPy만 쓴다(exec="browser").
실행: uv run --project labs --group torch python labs/deep-learning-pytorch/dashboards/cells.py
"""
# %% cell 1 — 손실 말고 무엇을 기록하는가: gradient norm, 파라미터 노름, 업데이트 비율
import torch
from torch import nn

def train_and_log(lr, steps=60, every=15):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    x = torch.randn(256, 8)                                        # (N, 8)
    y = x[:, :1] * 2 - x[:, 1:2] + 0.1 * torch.randn(256, 1)       # (N, 1)
    for step in range(1, steps + 1):
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters()])
        before = torch.cat([p.detach().flatten() for p in model.parameters()])
        opt.step()
        after = torch.cat([p.detach().flatten() for p in model.parameters()])
        if step % every == 0 or not torch.isfinite(loss):
            print(f"lr {lr:<4}  step {step:3d}  loss {loss.item():10.4g}  |g| {grads.norm():9.3g}  "
                  f"|θ| {after.norm():9.3g}  |Δθ|/|θ| {(after - before).norm() / before.norm():9.2e}")
        if not torch.isfinite(loss):
            break

train_and_log(0.05)
train_and_log(0.5, every=2)

# %% cell 2 — 학습 곡선의 네 가지 모양을 숫자로 읽기
import numpy as np

ep = np.arange(1, 51)
noise = lambda s: 0.01 * np.random.default_rng(s).normal(size=50)
curves = {
    "과적합":   (1.0 * np.exp(-ep / 8) + 0.05,       0.9 * np.exp(-ep / 8) + 0.25 + 0.004 * ep),
    "과소적합": (0.6 + 0.2 * np.exp(-ep / 5),       0.62 + 0.2 * np.exp(-ep / 5)),
    "학습률 큼": (0.4 + 0.3 * np.abs(np.sin(ep)),   0.45 + 0.3 * np.abs(np.sin(ep + 1))),
    "학습률 작음": (1.0 - 0.006 * ep,               1.02 - 0.006 * ep),
}
print("모양         마지막 train  val 최저 epoch  마지막 gap  마지막 10 epoch val 기울기  val 흔들림")
for i, (name, (tr, va)) in enumerate(curves.items()):
    tr, va = tr + noise(i), va + noise(i + 10)                           # (50,), (50,)
    slope = np.polyfit(ep[-10:], va[-10:], 1)[0]
    wiggle = np.std(np.diff(va))
    print(f"{name:10s} {tr[-1]:10.3f}  {va.argmin() + 1:10d}  {va[-1] - tr[-1]:12.3f}  {slope:20.4f}  {wiggle:12.3f}")

# %% cell 3 — smoothing이 숨기는 것: 지연, 꺾인 spike, 시작 부분의 치우침
rng = np.random.default_rng(0)
steps = np.arange(1, 1001)
base = 2.0 * np.exp(-steps / 150) + 0.3 + 0.0008 * np.maximum(steps - 800, 0)   # 800 step부터 상승
loss = base + 0.08 * rng.normal(size=1000)                               # (1000,) 기록된 원본
loss[600:605] += 3.0                                                     # 5 step짜리 spike

def ema(x, weight, debias):
    s, out = 0.0, np.empty_like(x)
    for i, v in enumerate(x):
        s = weight * s + (1 - weight) * v
        out[i] = s / (1 - weight ** (i + 1)) if debias else s
    return out

print(f"원본      : 상승 구간(900–1000)에서 기저 곡선보다 평균 {np.mean(base[899:] - loss[899:]):.3f} 낮게 보임")
for w in (0.9, 0.99):
    sm = ema(loss, w, debias=True)
    print(f"EMA {w:<4} : 상승 구간(900–1000)에서 기저 곡선보다 평균 {np.mean(base[899:] - sm[899:]):.3f} 낮게 보임, "
          f"spike 최대 {loss[595:615].max():.2f} → {sm[595:615].max():.2f}")
print(f"보정 없는 EMA의 첫 값 {ema(loss, 0.99, debias=False)[0]:.3f}, 원본 첫 값 {loss[0]:.3f}")

# %% cell 4 — 대시보드의 train loss와 다시 계산한 train loss
rng = np.random.default_rng(1)
X = rng.normal(size=(2000, 5))                                           # (N, D)
y = X @ np.array([1.0, -1.0, 0.5, 0.0, 2.0]) + 0.5 * rng.normal(size=2000)
w, lr, batch_losses = np.zeros(5), 0.02, []
for i in range(0, 2000, 20):                                             # 한 epoch, batch 100개
    Xb, yb = X[i:i + 20], y[i:i + 20]
    r = Xb @ w - yb
    batch_losses.append(np.mean(r ** 2))                                 # 업데이트 전 batch 손실
    w -= lr * 2 * Xb.T @ r / 20
print(f"epoch 동안 batch 손실의 평균(대시보드)  {np.mean(batch_losses):.3f}")
print(f"epoch 끝 모델로 다시 계산한 train 손실  {np.mean((X @ w - y) ** 2):.3f}")
print(f"100 step마다 찍은 batch 손실 하나        {batch_losses[-1]:.3f}  (마지막 batch)")
print(f"잡음 분산(도달 가능한 최소 손실)        {0.5 ** 2:.3f}")
