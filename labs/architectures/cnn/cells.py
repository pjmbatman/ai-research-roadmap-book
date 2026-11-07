"""18.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 4는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/architectures/cnn/cells.py
"""
# %% cell 1 — NumPy로 쓴 conv2d와 출력 크기 공식
import numpy as np
from scipy.signal import correlate2d

def out_size(n, k, s=1, p=0, d=1):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1

def conv2d(x, w, s=1, p=0, d=1):                     # x (C_in, H, W), w (C_out, C_in, k, k)
    C_out, C_in, k, _ = w.shape
    x = np.pad(x, ((0, 0), (p, p), (p, p)))          # zero padding
    H_out, W_out = out_size(x.shape[1], k, s, 0, d), out_size(x.shape[2], k, s, 0, d)
    y = np.zeros((C_out, H_out, W_out))              # (C_out, H_out, W_out)
    span = d * (k - 1) + 1                           # dilation을 반영한 kernel의 실제 폭
    for i in range(H_out):
        for j in range(W_out):
            patch = x[:, i * s:i * s + span:d, j * s:j * s + span:d]   # (C_in, k, k)
            y[:, i, j] = np.einsum("chw,ochw->o", patch, w)
    return y

rng = np.random.default_rng(0)
x = rng.normal(size=(3, 32, 32))                     # (C_in, H, W)
w = rng.normal(size=(8, 3, 5, 5))                    # (C_out, C_in, k, k)
for s, p, d in [(1, 0, 1), (1, 2, 1), (2, 2, 1), (1, 4, 2)]:
    y = conv2d(x, w, s, p, d)
    print(f"stride {s}, padding {p}, dilation {d} → {y.shape}  공식 H_out = {out_size(32, 5, s, p, d)}")
ref = sum(correlate2d(x[c], w[0, c], mode="valid") for c in range(3))   # 채널마다 상호상관 후 합
print("scipy correlate2d와 일치:", np.allclose(conv2d(x, w)[0], ref))

# %% cell 2 — stride는 이동 등변성을 깬다
def conv1d_circ(x, w, s=1):                          # x (T,), w (k,), 원형 경계
    y = np.array([w @ np.roll(x, -t)[:len(w)] for t in range(len(x))])   # stride 1 출력 (T,)
    return y[::s]                                    # stride s = stride 1 출력을 s칸마다 하나씩

x1, w1 = rng.normal(size=16), rng.normal(size=3)
for s in [1, 2]:
    for shift in [1, 2]:
        y_shifted_in = conv1d_circ(np.roll(x1, shift), w1, s)
        y = conv1d_circ(x1, w1, s)
        is_shift = any(np.allclose(y_shifted_in, np.roll(y, m)) for m in range(len(y)))
        print(f"stride {s}, 입력 {shift}칸 이동 → 출력이 이동만 된 것인가: {is_shift}")

# %% cell 3 — 이론 receptive field와 유효 receptive field
def receptive_field(layers):                         # layers: [(k, s, d), ...] 입력 쪽부터
    r, jump = 1, 1
    for k, s, d in layers:
        r += (k - 1) * d * jump                      # 이 층이 넓히는 폭(입력 칸 단위)
        jump *= s                                    # 이 층 출력의 한 칸 = 입력의 jump칸
    return r

print("3x3 다섯 층              :", receptive_field([(3, 1, 1)] * 5))
print("3x3 stride 2 다섯 층     :", receptive_field([(3, 2, 1)] * 5))
print("k=3 dilation 1,2,4,8,16 :", receptive_field([(3, 1, 2 ** i) for i in range(5)]))

print("층 수 L   이론 RF   중앙 95% 폭")                # 같은 평균 kernel을 L번 쌓은 선형 CNN
for L in [4, 16, 64]:
    h = np.array([1.0])
    for _ in range(L):
        h = np.convolve(h, np.ones(3) / 3)           # impulse response (2L+1,)
    m = next(m for m in range(L + 1) if h[L - m:L + m + 1].sum() >= 0.95)
    print(f"{L:7d}   {len(h):7d}   {2 * m + 1:11d}")

# %% cell 4 — PyTorch의 1D·2D·3D 합성곱과 shape
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
x1 = torch.randn(4, 2, 1000)                         # (B, C, T): 2채널 센서 시계열
x2 = torch.randn(4, 3, 64, 64)                       # (B, C, H, W): 영상
x3 = torch.randn(4, 1, 16, 32, 32)                   # (B, C, D, H, W): 3차원 장
conv1 = nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3)
conv2 = nn.Conv2d(3, 16, kernel_size=3, padding="same")
conv3 = nn.Conv3d(1, 8, kernel_size=3, stride=(1, 2, 2), padding=1)
for name, conv, x in [("1D", conv1, x1), ("2D", conv2, x2), ("3D", conv3, x3)]:
    n_param = sum(p.numel() for p in conv.parameters())
    print(f"{name}: {tuple(x.shape)} → {tuple(conv(x).shape)}  파라미터 {n_param}")

patches = F.pad(x1, (3, 3)).unfold(-1, 7, 2)          # (B, C_in, T_out, k): stride 2로 자른 창
manual = torch.einsum("bctk,ock->bot", patches, conv1.weight) + conv1.bias[:, None]
print("Conv1d = 창마다 가중합(상호상관):", torch.allclose(conv1(x1), manual, atol=1e-5))
