"""17.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 2·3은 NumPy만 쓰는 브라우저 셀이고(exec="browser"), 나머지는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/deep-learning-pytorch/training-loop/cells.py
"""
# %% cell 1 — 표준 학습 루프 전체
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
X = torch.rand(512, 2) * 4 - 2                              # (N, 2) 입력
y = torch.sin(X[:, :1]) * X[:, 1:] + 0.05 * torch.randn(512, 1)   # (N, 1)
train = DataLoader(TensorDataset(X[:384], y[:384]), batch_size=32, shuffle=True)
val_x, val_y = X[384:], y[384:]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 1)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
for epoch in range(30):
    model.train()
    for xb, yb in train:                                    # batch: (B, 2), (B, 1)
        xb, yb = xb.to(device), yb.to(device)               # device 이동
        pred = model(xb)                                    # forward: y_hat = f_theta(x)
        loss = nn.functional.mse_loss(pred, yb)             # loss: L = l(y_hat, y)
        opt.zero_grad()                                     # .grad를 비운다
        loss.backward()                                     # backward: g = dL/dtheta
        opt.step()                                          # update: theta <- Update(theta, g)
    if epoch % 10 == 9:
        model.eval()
        with torch.no_grad():                               # 평가: 그래프를 만들지 않는다
            val_mae = (model(val_x.to(device)) - val_y.to(device)).abs().mean().item()
        print(f"epoch {epoch + 1:2d}  마지막 batch 손실 {loss.item():.4f}  검증 MAE {val_mae:.4f}")

# %% cell 2 — 수식과 코드: 선형회귀 한 step을 NumPy로
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(8, 3))                                 # (B, D) batch 하나
y = X @ np.array([1.0, -2.0, 0.5]) + 0.3                    # (B,)
w, b, lr = np.zeros(3), 0.0, 0.1                            # theta = (w, b)
for step in range(3):
    y_hat = X @ w + b                                       # y_hat = f_theta(x)         (B,)
    loss = np.mean((y_hat - y) ** 2)                        # L = (1/B) sum (y_hat - y)^2
    r = 2 * (y_hat - y) / len(y)                            # dL/dy_hat                  (B,)
    g_w, g_b = X.T @ r, r.sum()                             # g = dL/dtheta (chain rule) (D,), ()
    w, b = w - lr * g_w, b - lr * g_b                       # theta <- theta - lr * g
    print(f"step {step}  L = {loss:.4f}  |g_w| = {np.linalg.norm(g_w):.4f}")
h = 1e-6                                                    # g_w[0]를 중앙차분으로 확인
L = lambda w0: np.mean((X @ np.r_[w0, w[1:]] + b - y) ** 2)
print("해석적 g_w[0] =", round(float(X[:, 0] @ (2 * (X @ w + b - y) / 8)), 6),
      " 수치미분 =", round((L(w[0] + h) - L(w[0] - h)) / (2 * h), 6))

# %% cell 3 — batch·step·epoch과 gradient accumulation
N, micro_B, accum = 1000, 32, 4                             # 표본 수, micro-batch, 누적 횟수
micro_per_epoch = int(np.ceil(N / micro_B))                 # DataLoader가 내는 batch 수
steps_per_epoch = int(np.ceil(micro_per_epoch / accum))     # optimizer.step() 횟수
print(f"micro-batch {micro_per_epoch}개/epoch, optimizer step {steps_per_epoch}회/epoch, "
      f"유효 batch {micro_B * accum}")
total_steps = 10 * steps_per_epoch                          # 10 epoch 동안의 스케줄 길이
cosine = lambda k: 0.5 * 1e-3 * (1 + np.cos(np.pi * min(k, total_steps) / total_steps))
k_right = 5 * steps_per_epoch                               # 5 epoch 뒤, step 단위로 센 경우
k_wrong = 5 * micro_per_epoch                               # micro-batch마다 scheduler.step()
print(f"5 epoch 뒤 학습률  올바름 {cosine(k_right):.2e}   micro-batch마다 호출 {cosine(k_wrong):.2e}")

# %% cell 4 — zero_grad 위치에 따른 차이
def run(order, steps=60):
    torch.manual_seed(0)
    Xt = torch.randn(256, 3)                                # (N, D)
    yt = Xt @ torch.tensor([1.0, -2.0, 0.5]) + 0.3          # (N,)
    m = nn.Linear(3, 1)
    o = torch.optim.SGD(m.parameters(), lr=0.05)
    for _ in range(steps):
        loss = nn.functional.mse_loss(m(Xt).squeeze(1), yt)
        if order == "올바름":
            o.zero_grad(); loss.backward(); o.step()
        elif order == "zero_grad 누락":
            loss.backward(); o.step()
        elif order == "backward 뒤 zero_grad":
            loss.backward(); o.zero_grad(); o.step()
    return loss.item(), [round(v, 3) for v in m.weight.detach().flatten().tolist()]

for order in ["올바름", "zero_grad 누락", "backward 뒤 zero_grad"]:
    final, wt = run(order)
    print(f"{order:20s} 최종 손실 {final:9.4f}  가중치 {wt}")

# %% cell 5 — mixed precision, gradient clipping, scheduler의 위치
import warnings

torch.manual_seed(0)
m = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
o = torch.optim.SGD(m.parameters(), lr=0.1)
sched = torch.optim.lr_scheduler.StepLR(o, step_size=1, gamma=0.5)
xb, yb = torch.randn(8, 4), torch.randn(8, 1)
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):   # CPU에서는 bfloat16
    pred = m(xb)
    loss = nn.functional.mse_loss(pred.float(), yb)
print("autocast 안  pred:", pred.dtype, " loss:", loss.dtype, " 파라미터:", m[0].weight.dtype)
o.zero_grad()
loss.backward()
total = nn.utils.clip_grad_norm_(m.parameters(), max_norm=0.5)   # backward 뒤, step 앞
after = torch.cat([p.grad.flatten() for p in m.parameters()]).norm()
print(f"clipping 전 |g| {total.item():.3f} -> 후 {after.item():.3f}")
o.step()
sched.step()                                                # optimizer.step() 뒤에
print("학습률:", o.param_groups[0]["lr"])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    o2 = torch.optim.SGD(m.parameters(), lr=0.1)
    torch.optim.lr_scheduler.StepLR(o2, step_size=1).step()   # step 전에 호출하면
print("경고:", any("before `optimizer.step()`" in str(c.message) for c in caught))
