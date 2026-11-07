"""17.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 1, 3, 4는 PyTorch가 필요하다(exec="python"). cell 2, 5는 NumPy만 쓴다(exec="browser").
실행: uv run --project labs --group torch python labs/deep-learning-pytorch/evaluation-and-checkpoints/cells.py
"""
# %% cell 1 — train/eval 모드와 gradient 모드는 서로 다른 스위치다
import torch
from torch import nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(),
                      nn.Dropout(0.5), nn.Linear(8, 1))
x = torch.randn(16, 4)                                   # (B, D)
bn = model[1]

with torch.no_grad():                                    # gradient 기록만 끈다. 모드는 train 그대로
    y1, y2 = model(x), model(x)                          # (B, 1)
print("no_grad + train: 두 번 호출이 같은가 ", torch.equal(y1, y2))
print("no_grad + train: running_mean이 바뀌었는가", bool(bn.running_mean.abs().sum() > 0))

model.eval()                                             # dropout 끔, BN은 저장된 running 통계 사용
y3, y4 = model(x), model(x)
print("eval: 두 번 호출이 같은가          ", torch.equal(y3, y4))
print("eval: 출력이 그래프에 기록되는가   ", y3.requires_grad)
print("eval: 표본 0의 출력이 batch와 무관한가", torch.equal(model(x[:4])[0], y3[0]))

with torch.inference_mode():                             # 기록도 끄고 버전 추적도 생략
    y5 = model(x)
print("inference_mode: requires_grad", y5.requires_grad, " 값이 eval 출력과 같은가", torch.equal(y5, y3.detach()))

# %% cell 2 — batch 평균의 평균과 mask가 있는 집계
import numpy as np

rng = np.random.default_rng(0)
abs_err = np.sort(np.abs(rng.normal(size=1000)))         # (N,) 평가 데이터가 난이도 순으로 놓인 경우
batches = [abs_err[i:i + 64] for i in range(0, 1000, 64)]
print("batch 수", len(batches), " 마지막 batch 크기", len(batches[-1]))

total_sum, total_n = 0.0, 0
for b in batches:                                        # 합과 개수를 따로 누적한다
    total_sum, total_n = total_sum + b.sum(), total_n + b.size
print(f"batch 평균의 평균 {np.mean([b.mean() for b in batches]):.4f}")
print(f"합 / 개수         {total_sum / total_n:.4f}   (전체 한 번에 {abs_err.mean():.4f})")

# 길이가 다른 시계열 3개를 길이 5로 padding했다. mask = 1이 실제 관측
err = np.array([[1., 1., 1., 1., 1.],
                [2., 2., 0., 0., 0.],
                [4., 0., 0., 0., 0.]])                   # (B, T)
mask = (np.arange(5) < np.array([[5], [2], [1]])).astype(float)   # (B, T)
print(f"padding 포함 평균       {err.mean():.4f}")
print(f"관측 단위 평균          {(err * mask).sum() / mask.sum():.4f}")
print(f"시계열 단위 평균의 평균 {((err * mask).sum(1) / mask.sum(1)).mean():.4f}")

# %% cell 3 — 작은 학습: 모델·optimizer·scheduler·RNG가 모두 결과에 관여한다
import io

def make(seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    return model, opt, sched

def train(model, opt, sched, steps):
    model.train()
    for _ in range(steps):
        x = torch.randn(32, 3)                           # (B, 3) 데이터 표집도 전역 RNG를 쓴다
        y = x.sum(1, keepdim=True)                       # (B, 1)
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

def flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

m, o, s = make(seed=0)
train(m, o, s, 40)
straight = flat(m)                                       # 끊김 없이 40 step
print("파라미터 수", straight.numel(), " 마지막 학습률", s.get_last_lr()[0])

# %% cell 4 — 20 step에서 저장하고 새 객체로 재개한다
def resume(restore):
    m, o, s = make(seed=0)
    train(m, o, s, 20)
    buf = io.BytesIO()
    torch.save({"model": m.state_dict(), "optimizer": o.state_dict(), "scheduler": s.state_dict(),
                "rng": torch.get_rng_state(), "step": 20}, buf)
    buf.seek(0)
    ckpt = torch.load(buf)
    m, o, s = make(seed=123)                             # 새 프로세스라고 생각한다
    m.load_state_dict(ckpt["model"])
    if "optimizer" in restore: o.load_state_dict(ckpt["optimizer"])
    if "scheduler" in restore: s.load_state_dict(ckpt["scheduler"])
    if "rng" in restore: torch.set_rng_state(ckpt["rng"])
    train(m, o, s, 40 - ckpt["step"])
    return flat(m)

full = {"optimizer", "scheduler", "rng"}
for name, restore in [("전부 복원", full), ("optimizer 빠짐", full - {"optimizer"}),
                      ("scheduler 빠짐", full - {"scheduler"}), ("RNG 빠짐", full - {"rng"})]:
    p = resume(restore)
    print(f"{name:14s} 일치 {torch.equal(p, straight)!s:5s}  최대 차이 {(p - straight).abs().max():.2e}")

# %% cell 5 — best checkpoint 선택과 early stopping
rng = np.random.default_rng(1)
epochs = np.arange(1, 61)
true_val = 0.5 + 0.5 * np.exp(-epochs / 8) + 0.004 * epochs   # (60,) 내려갔다 다시 오르는 기저 곡선
val = true_val + rng.normal(0, 0.02, size=60)                  # 관측한 검증 손실
test = true_val + rng.normal(0, 0.02, size=60)                 # 같은 epoch의 독립된 test 손실

best, best_epoch, wait, patience, min_delta = np.inf, 0, 0, 5, 1e-3
for e, v in zip(epochs, val):
    if v < best - min_delta:                             # 의미 있게 좋아졌을 때만 best 갱신
        best, best_epoch, wait = v, e, 0
    else:
        wait += 1
        if wait >= patience:
            break
print(f"정지 epoch {e}, best epoch {best_epoch}, best의 검증 손실 {best:.3f}")
print(f"best epoch의 test 손실 {test[best_epoch - 1]:.3f}, 기저 곡선 {true_val[best_epoch - 1]:.3f}")
print(f"기저 곡선이 가장 낮은 epoch {epochs[true_val.argmin()]} (값 {true_val.min():.3f})")
