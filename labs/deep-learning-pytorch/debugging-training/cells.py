"""17.9 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 1, 3은 PyTorch가 필요하다(exec="python"). cell 2, 4는 NumPy만 쓴다(exec="browser").
실행: uv run --project labs --group torch python labs/deep-learning-pytorch/debugging-training/cells.py
"""
# %% cell 1 — 초기 손실과 batch 하나 과적합: 무엇을 잡고 무엇을 놓치는가
import math

import torch
from torch import nn

torch.manual_seed(0)
x = torch.randn(16, 10)                                   # (B, D) batch 하나
y = torch.randint(0, 5, (16,))                            # (B,) class 5개

def overfit(head, labels, steps=300):
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 5), *head)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(steps + 1):
        loss = nn.functional.cross_entropy(model(x), labels)
        first = loss.item() if step == 0 else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    return first, loss.item()

print(f"기대 초기 손실 ln 5 = {math.log(5):.3f}")
for name, head, labels in [("정상", [], y), ("softmax 두 번", [nn.Softmax(dim=1)], y),
                           ("label 무작위", [], y[torch.randperm(16)])]:
    first, last = overfit(head, labels)
    print(f"{name:12s} 초기 손실 {first:.3f}   300 step 뒤 {last:.4f}")

# %% cell 2 — 데이터부터 확인한다: 학습 전에 돌리는 점검
import numpy as np

rng = np.random.default_rng(2)
X = rng.normal([300.0, 1.0, 0.5, 7.0], [10.0, 0.1, 0.2, 0.0], size=(500, 4))   # (N, 4)
X[rng.choice(500, 3, replace=False), 1] = np.nan          # 결측 3개
y = (X[:, 0] > 300).astype(int)
y[:40] = 1                                                # 앞쪽 40개 label이 잘못 붙음
train, test = X[:400], np.vstack([X[400:490], X[:10]])    # test에 train 표본 10개가 섞임

print("특징별 평균  ", np.round(np.nanmean(X, 0), 3))
print("특징별 표준편차", np.round(np.nanstd(X, 0), 3))
print("NaN 개수     ", np.isnan(X).sum(0))
print("상수 특징     ", np.where(np.nanstd(X, 0) == 0)[0])
print("label 비율    ", np.bincount(y) / len(y))
dup = {r.tobytes() for r in train} & {r.tobytes() for r in test}
print("train과 test에 함께 있는 행", len(dup))
print("앞 50개와 나머지의 label 1 비율", y[:50].mean().round(3), y[50:].mean().round(3))

# %% cell 3 — 손실은 유한한데 gradient는 NaN: 처음 비유한 값이 나온 곳 찾기
import warnings

class LogFeature(nn.Module):
    """농도 같은 양수 특징의 로그를 취한다(0이 들어오면 -inf)."""
    def forward(self, x):
        return torch.cat([torch.log(x[:, :1]), x[:, 1:]], dim=1)

torch.manual_seed(0)
model = nn.Sequential(LogFeature(), nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1))
x = torch.rand(6, 3)                                      # (B, 3)
x[4, 0] = 0.0                                             # 농도 0인 표본 하나
first_bad = []
def find_nonfinite(module, inputs, output):
    if not first_bad and not torch.isfinite(output).all():
        rows = (~torch.isfinite(output)).any(1).nonzero().flatten().tolist()
        first_bad.append(f"{module.__class__.__name__} (표본 {rows})")
for m in model:
    m.register_forward_hook(find_nonfinite)
loss = model(x).pow(2).mean()
loss.backward()
print(f"손실 {loss.item():.4f}, 유한한가 {torch.isfinite(loss).item()}")
print("처음 비유한 출력을 낸 모듈:", first_bad[0])
print("NaN gradient:", [n for n, p in model.named_parameters() if not torch.isfinite(p.grad).all()])

w = torch.zeros(3, requires_grad=True)                    # backward에서만 생기는 NaN
with warnings.catch_warnings(record=True) as caught, torch.autograd.set_detect_anomaly(True):
    try:
        torch.sqrt((w ** 2).sum()).backward()             # |w|를 w = 0에서 미분
    except RuntimeError as e:
        print("anomaly mode:", str(e))

# %% cell 4 — 활성값 통계: 초기화 스케일이 깊은 층에서 만드는 차이
rng = np.random.default_rng(0)
x0 = rng.normal(size=(512, 256))                          # (N, D) 표준화된 입력
for name, std in [("std 1", 1.0), ("std 1/sqrt(D)", 1 / np.sqrt(256)), ("std 0.01", 0.01)]:
    a, rows = x0, []
    for layer in range(1, 11):
        W = rng.normal(0, std, size=(256, 256))           # (D, D)
        a = np.tanh(a @ W)                                # (N, D)
        if layer in (1, 5, 10):
            rows.append(f"층 {layer:2d}: std {a.std():.3g}, 포화 {np.mean(np.abs(a) > 0.99):.0%}")
    print(f"{name:14s} | " + " | ".join(rows))
