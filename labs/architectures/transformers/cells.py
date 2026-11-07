"""18.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 5는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/architectures/transformers/cells.py
"""
# %% cell 1 — scaled dot-product attention과 √D 스케일
import numpy as np

rng = np.random.default_rng(0)

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)          # 최댓값을 빼도 결과는 같고 overflow를 막는다
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):                   # Q (..., T, D), K (..., S, D), V (..., S, Dv)
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])   # (..., T, S)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)     # mask가 False인 칸은 가중치가 정확히 0
    w = softmax(scores)                              # (..., T, S), 행마다 합 1
    return w @ V, w                                  # (..., T, Dv), (..., T, S)

print("   D   Var(q·k)   최대 가중치 평균: 스케일 없음   √D로 나눔")
for D in [4, 64, 512]:
    Q, K = rng.normal(size=(16, D)), rng.normal(size=(16, D))   # (T, D), (S, D)
    raw = Q @ K.T                                    # (T, S)
    print(f"{D:4d}  {raw.var():9.1f}   {softmax(raw).max(-1).mean():22.3f}   {softmax(raw / np.sqrt(D)).max(-1).mean():9.3f}")

# %% cell 2 — multi-head self-attention은 순열 등변이다
def mha(X, n_heads=4, mask=None):                    # X (T, D) → (T, D)
    T, D = X.shape
    split = lambda Z: Z.reshape(T, n_heads, D // n_heads).transpose(1, 0, 2)   # (T, D) → (H, T, Dh)
    out, w = attention(split(X @ Wq), split(X @ Wk), split(X @ Wv), mask)     # (H, T, Dh), (H, T, T)
    return out.transpose(1, 0, 2).reshape(T, D) @ Wo, w                       # head를 다시 이어 붙임

def sinusoidal(T, D):                                # 절대 위치 인코딩 (T, D)
    angle = np.arange(T)[:, None] / 10000 ** (np.arange(0, D, 2)[None, :] / D)   # (T, D/2)
    pe = np.zeros((T, D))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
    return pe

D, T = 32, 10
Wq, Wk, Wv, Wo = (rng.normal(size=(D, D)) / np.sqrt(D) for _ in range(4))
X = rng.normal(size=(T, D))                          # 토큰 10개 (T, D)
P = np.eye(T)[rng.permutation(T)]                    # 순서를 섞는 치환 행렬 (T, T)
PE = sinusoidal(T, D)
print("위치 정보 없음  f(PX) = P f(X)            :", np.allclose(mha(P @ X)[0], P @ mha(X)[0]))
print("위치 인코딩 더함 f(PX + PE) = P f(X + PE) :", np.allclose(mha(P @ X + PE)[0], P @ mha(X + PE)[0]))
print("평균 pooling 후 순열 불변                  :", np.allclose(mha(P @ X)[0].mean(0), mha(X)[0].mean(0)))

# %% cell 3 — RoPE: 회전한 query·key의 내적은 상대 위치에만 의존한다
def rope(x, pos, base=10000.0):                      # x (D,), 성분 쌍 (2i, 2i+1)을 pos·θ_i만큼 회전
    theta = pos * base ** (-np.arange(0, len(x), 2) / len(x))   # (D/2,)
    c, s = np.cos(theta), np.sin(theta)
    out = np.empty_like(x)
    out[0::2] = x[0::2] * c - x[1::2] * s
    out[1::2] = x[0::2] * s + x[1::2] * c
    return out

q, k = rng.normal(size=D), rng.normal(size=D)
for m, n in [(3, 1), (10, 8), (52, 50), (52, 49)]:
    print(f"query 위치 {m:2d}, key 위치 {n:2d} (차이 {m - n}): q·k = {rope(q, m) @ rope(k, n):8.4f}")

# %% cell 4 — causal mask와 padding mask
T = 6
Xp = rng.normal(size=(T, D))
Xp[4:] = 0.0                                         # 마지막 두 칸은 PAD (실제 토큰 4개)
causal = np.tril(np.ones((T, T), dtype=bool))        # query t는 key s ≤ t만 본다 (T, T)
pad = np.ones((T, T), dtype=bool)
pad[:, 4:] = False                                   # 모든 query가 PAD key를 보지 않는다 (T, T)

Xf = Xp.copy()
Xf[2] += 1.0                                         # 시점 2의 토큰만 바꾼다
changed = np.abs(mha(Xf, mask=causal)[0] - mha(Xp, mask=causal)[0]).max(axis=1) > 0
print("causal mask: 시점 2를 바꾸면 출력이 바뀌는 시점:", np.flatnonzero(changed))

Xq = Xp.copy()
Xq[4:] = rng.normal(size=(2, D))                     # PAD 칸에 다른 값이 들어 있다면
for name, m in [("mask 없음    ", None), ("padding mask", pad)]:
    d = np.abs(mha(Xq, mask=m)[0][:4] - mha(Xp, mask=m)[0][:4]).max()
    print(f"{name}: PAD 값을 바꿨을 때 실제 토큰 출력의 최대 변화 =", "0" if d == 0 else f"{d:.3f}")

# %% cell 5 — 직접 구현한 multi-head attention과 PyTorch 구현 비교
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
B, T, D, H = 2, 7, 32, 4
x = torch.randn(B, T, D)                             # (B, T, D)
ref = nn.MultiheadAttention(D, H, batch_first=True)
W_q, W_k, W_v = ref.in_proj_weight.chunk(3)           # 각 (D, D); PyTorch 선형층은 x @ W.T + b
b_q, b_k, b_v = ref.in_proj_bias.chunk(3)

def my_mha(x, causal=False):                         # (B, T, D) → (B, T, D)
    Dh = D // H
    split = lambda z: z.view(B, T, H, Dh).transpose(1, 2)        # (B, H, T, Dh)
    q, k, v = split(x @ W_q.T + b_q), split(x @ W_k.T + b_k), split(x @ W_v.T + b_v)
    s = q @ k.transpose(-1, -2) / Dh ** 0.5                      # (B, H, T, T)
    if causal:
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)     # head를 다시 이어 붙임 (B, T, D)
    return ref.out_proj(o)

future = torch.ones(T, T, dtype=torch.bool).triu(1)  # True = 가린다 (PyTorch의 bool mask 규칙)
for causal in [False, True]:
    y_ref, _ = ref(x, x, x, attn_mask=future if causal else None, need_weights=False)
    print(f"causal={causal!s:5}  nn.MultiheadAttention과 최대 차 < 1e-5:", (my_mha(x, causal) - y_ref).abs().max().item() < 1e-5)
q = k = v = torch.randn(B, H, T, 8)
manual = ((q @ k.transpose(-1, -2)) / 8 ** 0.5).masked_fill(future, float("-inf")).softmax(-1) @ v
print("F.scaled_dot_product_attention(is_causal=True)과 일치:",
      torch.allclose(F.scaled_dot_product_attention(q, k, v, is_causal=True), manual, atol=1e-6))

# %% cell 6 — 이차 비용과 KV cache
for T_ in [1_024, 8_192, 65_536]:
    gb = T_ ** 2 * 16 * 4 / 1e9                      # head 16개, float32, 한 층·한 시퀀스의 점수 행렬
    print(f"T = {T_:6d}: 점수 행렬 {gb:8.2f} GB")

T = 8
Xs = rng.normal(size=(T, D))                         # (T, D)
full = mha(Xs, mask=np.tril(np.ones((T, T), dtype=bool)))[0]   # 한 번에 계산한 causal 출력 (T, D)
heads = lambda Z: Z.reshape(len(Z), 4, D // 4).transpose(1, 0, 2)   # (t, D) → (H, t, Dh)
K_cache, V_cache, steps = [], [], []
for t in range(T):                                   # 토큰을 하나씩 생성하는 상황
    K_cache.append(Xs[t] @ Wk)                       # 새 토큰의 key·value만 계산해 덧붙인다
    V_cache.append(Xs[t] @ Wv)
    q = heads((Xs[t] @ Wq)[None])                    # (H, 1, Dh)
    o, _ = attention(q, heads(np.array(K_cache)), heads(np.array(V_cache)))   # 과거 전체를 본다
    steps.append(o.transpose(1, 0, 2).reshape(D) @ Wo)
print("KV cache로 한 step씩 = 전체 causal 계산:", np.allclose(np.array(steps), full))
print("cache 크기 (key + value, 원소 수):", 2 * len(K_cache) * D)
