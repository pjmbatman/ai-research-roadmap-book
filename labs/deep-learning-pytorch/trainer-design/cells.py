"""17.7 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 1, 2는 NumPy만 쓴다(exec="browser"). cell 3, 4는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/deep-learning-pytorch/trainer-design/cells.py
"""
# %% cell 1 — 책임만 가진 작은 Trainer: 과제는 step 함수로, 부가 기능은 callback으로
import numpy as np

class Trainer:
    def __init__(self, step_fn, eval_fn, callbacks=(), max_epochs=100):
        self.step_fn, self.eval_fn = step_fn, eval_fn          # 과제별 코드는 밖에서 받는다
        self.callbacks, self.max_epochs = list(callbacks), max_epochs
        self.should_stop, self.epoch = False, 0

    def _emit(self, event, **info):
        for cb in self.callbacks:
            getattr(cb, event, lambda trainer, **kw: None)(self, **info)

    def fit(self, params, batches):
        for self.epoch in range(1, self.max_epochs + 1):
            for batch in batches:
                params, loss = self.step_fn(params, batch)     # forward·backward·update는 과제가 정한다
            self._emit("on_epoch_end", train_loss=loss)
            self._emit("on_eval_end", metric=self.eval_fn(params))
            if self.should_stop:
                break
        return params

# %% cell 2 — 같은 Trainer로 손실이 다른 두 과제를 학습한다
class EarlyStopping:
    def __init__(self, patience, min_delta=1e-3):
        self.patience, self.min_delta, self.best, self.wait = patience, min_delta, np.inf, 0
    def on_eval_end(self, trainer, metric):
        improved = metric < self.best - self.min_delta
        self.best, self.wait = (metric, 0) if improved else (self.best, self.wait + 1)
        trainer.should_stop = self.wait >= self.patience

class History:
    def __init__(self):
        self.rows = []
    def on_eval_end(self, trainer, metric):
        self.rows.append((trainer.epoch, round(float(metric), 4)))

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))                                  # (N, D)
y = X @ np.array([1.0, -2.0, 0.5]) + rng.standard_t(df=2, size=200) * 0.3   # 두꺼운 꼬리 잡음
batches = [(X[i:i + 20], y[i:i + 20]) for i in range(0, 160, 20)]
Xv, yv = X[160:], y[160:]

def mse_step(w, b, lr=0.05):
    r = b[0] @ w - b[1]
    return w - lr * 2 * b[0].T @ r / len(r), np.mean(r ** 2)
def huber_step(w, b, lr=0.05, delta=1.0):
    r = b[0] @ w - b[1]
    return w - lr * b[0].T @ np.clip(r, -delta, delta) / len(r), np.mean(np.minimum(r ** 2 / 2, delta * (np.abs(r) - delta / 2)))
val_mae = lambda w: np.mean(np.abs(Xv @ w - yv))

for name, step in [("MSE", mse_step), ("Huber", huber_step)]:
    hist = History()
    w = Trainer(step, val_mae, [EarlyStopping(patience=3), hist], max_epochs=50).fit(np.zeros(3), batches)
    print(f"{name:5s} 멈춘 epoch {hist.rows[-1][0]:2d}  검증 MAE {hist.rows[-1][1]:.4f}  w = {np.round(w, 3)}")

# %% cell 3 — PyTorch: trainer는 루프와 순서만, 과제는 step 함수가 맡는다
import torch
from torch import nn

def fit(model, opt, loader, step_fn, max_epochs):
    for epoch in range(max_epochs):
        model.train()
        for batch in loader:
            loss = step_fn(model, batch)                       # 과제별: forward와 loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

def regression_step(model, batch):
    x, y = batch
    return nn.functional.mse_loss(model(x), y)

def plain_loop(model, opt, loader, max_epochs):                # 17.5장의 루프를 그대로 쓴 것
    for epoch in range(max_epochs):
        model.train()
        for x, y in loader:
            loss = nn.functional.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

def setup(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(64, 3)                                     # (N, 3)
    y = torch.sin(x[:, :1]) + x[:, 1:2] * x[:, 2:]              # (N, 1)
    data = list(zip(x.split(16), y.split(16)))                 # batch 4개
    model = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1))
    return model, torch.optim.Adam(model.parameters(), lr=1e-2), data

a = fit(*setup(), regression_step, max_epochs=200)
b = plain_loop(*setup(), max_epochs=200)
print("trainer와 직접 쓴 루프의 결과가 같은가:",
      all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters())))

# %% cell 4 — 이해하지 못한 기본값: Adam을 AdamW로 바꾸기만 했을 때
for opt_cls in (torch.optim.Adam, torch.optim.AdamW):
    model, _, data = setup()
    opt = opt_cls(model.parameters(), lr=1e-2)                 # 다른 인자는 주지 않았다
    fit(model, opt, data, regression_step, max_epochs=200)
    norm = torch.cat([p.detach().flatten() for p in model.parameters()]).norm()
    loss = sum(regression_step(model, batch).item() for batch in data) / len(data)
    print(f"{opt_cls.__name__:5s} weight_decay 기본값 {opt.defaults['weight_decay']:<5}  "
          f"파라미터 노름 {norm:.4f}  학습 손실 {loss:.4f}")
