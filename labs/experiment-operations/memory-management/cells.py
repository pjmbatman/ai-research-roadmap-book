"""22.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 2만 PyTorch가 필요하고(exec="python"), 나머지는 NumPy만 쓰는 브라우저 셀이다(exec="browser").
실행: uv run --project labs --group torch python labs/experiment-operations/memory-management/cells.py
"""
# %% cell 1 — 파라미터 하나당 byte로 세는 정적 메모리
import numpy as np

def mlp_params(widths):
    return sum(i * o + o for i, o in zip(widths[:-1], widths[1:]))   # 가중치 + bias

recipes = {                      # 파라미터 하나당 byte: (파라미터, gradient, optimizer 상태)
    "fp32 + SGD":              (4, 4, 0),
    "fp32 + SGD(momentum)":    (4, 4, 4),
    "fp32 + Adam":             (4, 4, 8),   # m, v
    "autocast(bf16) + Adam":   (4, 4, 8),   # 파라미터·gradient는 fp32 그대로
    "bf16 + fp32 사본 + Adam":  (2, 2, 12),  # 사본 4 + m 4 + v 4
}
n = mlp_params([1024, 4096, 4096, 4096, 1024])
print(f"파라미터 {n:,}개")
for name, (p, g, s) in recipes.items():
    print(f"{name:22s} {p + g + s:2d} byte/파라미터 → {n * (p + g + s) / 2**30:5.2f} GiB")
print(f"70억 파라미터 × 16 byte = {7e9 * 16 / 2**30:.0f} GiB")

# %% cell 2 — PyTorch로 측정: 상태 텐서와 backward용 저장 텐서
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

torch.manual_seed(0)
widths = [256, 1024, 1024, 1024, 256]
layers = [m for i, o in zip(widths[:-1], widths[1:]) for m in (nn.Linear(i, o), nn.ReLU())]
model = nn.Sequential(*layers[:-1])                       # 마지막 ReLU는 뺀다
param_ptrs = {p.data_ptr() for p in model.parameters()}

def saved_bytes(fn, x):
    """forward가 backward를 위해 저장하는 텐서(파라미터 제외)의 byte 수. 같은 텐서는 한 번만 센다."""
    seen = {}
    def pack(t):
        if t.data_ptr() not in param_ptrs:
            seen[t.data_ptr()] = t.numel() * t.element_size()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = fn(x).sum()
    loss.backward()
    return sum(seen.values())

x = torch.randn(64, widths[0])                            # (B, D)
ckpt = lambda h: checkpoint(model[4:], checkpoint(model[:4], h, use_reentrant=False), use_reentrant=False)
print("샘플당 저장 byte: 그대로", saved_bytes(model, x) // 64, "/ checkpointing", saved_bytes(ckpt, x) // 64)
print("식 4 × (Linear 입력 폭의 합) =", 4 * sum(widths[:-1]))
opt = torch.optim.Adam(model.parameters())
opt.step()
nbytes = lambda ts: sum(t.numel() * t.element_size() for t in ts)
print("파라미터", nbytes(model.parameters()), " gradient", nbytes(p.grad for p in model.parameters()),
      " Adam 상태", nbytes(v for s in opt.state.values() for v in s.values()))

# %% cell 3 — activation checkpointing: 구간 길이에 따른 메모리와 재계산
n_layers = 48                                    # 같은 크기의 층 48개
print(" 구간 길이 k   저장하는 층 수   추가 계산(fwd+bwd 대비)")
for k in [1, 2, 4, 6, 8, 16, 48]:
    if k == 1:
        stored, extra = n_layers, 0.0            # checkpointing 없음
    else:
        stored = int(np.ceil(n_layers / k)) + k  # 구간 경계 + backward 때 다시 만든 한 구간
        extra = (1 - 1 / k) / 3                  # 다시 하는 forward ≈ (1 - 1/k)회, backward ≈ forward 2회
    print(f"{k:8d} {stored:14d} {extra:16.0%}")

# %% cell 4 — 메모리 예산에서 micro-batch와 accumulation 정하기
gpu_gib = 24.0                                   # 가상의 GPU
static_gib = 350e6 * 16 / 2**30                  # 3.5억 파라미터, fp32 + Adam
reserve_gib = 2.0                                # CUDA context·단편화 여유(가정)
act_per_sample_gib = {"fp32": 0.40, "autocast bf16": 0.22, "bf16 + checkpointing": 0.06}  # 측정했다고 가정
target = 256                                     # 목표 유효 batch
print(f"정적 메모리 {static_gib:.2f} GiB, 활성값에 쓸 수 있는 메모리 {gpu_gib - static_gib - reserve_gib:.2f} GiB")
for name, a in act_per_sample_gib.items():
    max_b = int((gpu_gib - static_gib - reserve_gib) // a)
    micro = max(d for d in range(1, max_b + 1) if target % d == 0)   # target의 약수 중 가장 큰 것
    print(f"{name:22s} 최대 {max_b:3d} → micro-batch {micro:3d} × 누적 {target // micro:2d}")

# %% cell 5 — 8-bit optimizer 상태: 블록별 absmax 양자화
rng = np.random.default_rng(0)
m = rng.normal(scale=1e-3, size=4096).astype(np.float32)   # Adam 1차 moment 흉내, (4096,)
m[::512] *= 100                                            # 드문 큰 값(outlier) 8개

def absmax_int8(x, block):
    xb = x.reshape(-1, block)                              # (블록 수, block)
    scale = np.abs(xb).max(axis=1, keepdims=True) / 127    # 블록마다 fp32 scale 하나
    q = np.round(xb / scale).astype(np.int8)               # -127..127
    return (q * scale).reshape(x.shape), 1 + 4 / block     # 복원값, 원소당 byte

for block in [4096, 256, 64]:
    deq, bpe = absmax_int8(m, block)
    rel = np.linalg.norm(deq - m) / np.linalg.norm(m)
    print(f"블록 {block:4d}: 원소당 {bpe:.3f} byte  상대오차 {rel:.4f}  0이 된 원소 {np.mean(deq == 0):.1%}")
