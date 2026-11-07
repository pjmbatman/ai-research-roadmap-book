"""19.9 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 5는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/generative-and-foundation-models/foundation-models/cells.py
"""
# %% cell 1 — 파라미터 수와 미세조정 메모리: 무엇을 학습하는가
import numpy as np

L, d, V, r = 24, 1024, 32000, 8                    # 가상의 Transformer 구성: 층, 폭, 어휘, LoRA rank
attn = 4 * (d * d + d)                             # Q, K, V, O 투영 (가중치 + bias)
mlp = (d * 4 * d + 4 * d) + (4 * d * d + d)        # 폭 4d의 MLP 두 층
norm = 2 * 2 * d                                   # LayerNorm 2개 (scale, shift)
total = L * (attn + mlp + norm) + V * d            # + token embedding
lora = L * 2 * r * (d + d)                         # Q, V 투영에만 rank r 갱신 B A: (d×r) + (r×d)
print(f"전체 파라미터       {total / 1e6:8.1f} M   (12·L·d² 근사 {12 * L * d * d / 1e6:.1f} M + embedding {V * d / 1e6:.1f} M)")
print(f"LoRA 학습 파라미터  {lora / 1e6:8.3f} M   (전체의 {100 * lora / total:.3f}%)")
GB = 1024**3
full_ft = total * (4 + 4 + 8)                      # fp32 가중치 + gradient + Adam 1·2차 모멘트 (byte)
lora_ft = total * 4 + lora * (4 + 4 + 8)           # 고정 가중치 + LoRA 파라미터의 학습 상태
print(f"학습 상태 메모리(activation 제외)  full {full_ft / GB:.1f} GB,  LoRA {lora_ft / GB:.2f} GB")

# %% cell 2 — LoRA: 저랭크 갱신의 적합과 병합
rng = np.random.default_rng(0)
d_in, d_out = 64, 64
W0 = rng.normal(size=(d_out, d_in)) / np.sqrt(d_in)                # 사전학습된 고정 가중치
dW_true = rng.normal(size=(d_out, 2)) @ rng.normal(size=(2, d_in)) / np.sqrt(d_in)   # 과제의 변화: rank 2
dW_true += 0.02 * rng.normal(size=(d_out, d_in))                   # + 작은 full-rank 성분

def task(n):                                                       # 입력 (n, d_in), 출력 (n, d_out)
    X = rng.normal(size=(n, d_in))
    return X, X @ (W0 + dW_true).T + 0.5 * rng.normal(size=(n, d_out))

def fit_delta(X, Y, rank=None):                                    # W0는 고정, 잔차에 ΔW를 맞춘다
    C = np.linalg.lstsq(X, Y - X @ W0.T, rcond=None)[0]            # 최소제곱 ΔWᵀ (d_in, d_out)
    if rank is None:
        return C.T                                                 # full fine-tuning
    _, _, Vt = np.linalg.svd(X @ C, full_matrices=False)           # reduced-rank 회귀
    return (C @ Vt[:rank].T @ Vt[:rank]).T                         # rank-r ΔW = B A

X_te, Y_te = task(5000)
print("학습 표본   rank 1   rank 2   rank 8   full(4096개)   ΔW=0(사전학습 그대로)")
for n in [100, 2000]:
    X, Y = task(n)
    err = lambda dW: np.mean((Y_te - X_te @ (W0 + dW).T) ** 2)
    print(f"{n:6d}    " + "  ".join(f"{err(fit_delta(X, Y, r)):6.3f}" for r in [1, 2, 8]) +
          f"     {err(fit_delta(X, Y)):6.3f}          {err(0):6.3f}")

U, s, Vt = np.linalg.svd(fit_delta(X, Y, 2))                       # ΔW = B A로 인수분해
B, A = U[:, :2] * s[:2], Vt[:2]                                    # (d_out, 2), (2, d_in): 2·(64+64) = 256개
x = rng.normal(size=d_in)
print("병합 W0 + BA의 출력 = 두 경로 출력:", np.allclose((W0 + B @ A) @ x, W0 @ x + B @ (A @ x)))

# %% cell 3 — 전이의 효과와 부정적 전이: 원천 과제와 얼마나 닮았는가
p = 50
w_src = rng.normal(size=p)                                         # 원천 과제에서 배운 가중치 (p,)
targets = {"닮은 과제": w_src + 0.3 * rng.normal(size=p),          # 목표 = 원천 + 작은 차이
           "무관한 과제": rng.normal(size=p)}                       # 목표가 원천과 독립

def fit(X, y, prior, lam=20.0):                                    # min ‖y - Xw‖² + λ‖w - prior‖²
    return prior + np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ (y - X @ prior))

def r2(w, w_t, X_te):
    y_te = X_te @ w_t + rng.normal(size=len(X_te))
    return 1 - np.mean((y_te - X_te @ w) ** 2) / np.var(y_te)

X_te = rng.normal(size=(4000, p))
for name, w_t in targets.items():
    print(f"[{name}]  학습 표본   처음부터   미세조정(원천에서 출발)   고정 특징 x·w_src + 선형 head")
    for n in [10, 40, 200]:
        res = []
        for rep in range(20):                                      # 학습 표본을 바꿔 20번 평균
            X = rng.normal(size=(n, p)); y = X @ w_t + rng.normal(size=n)
            f = X @ w_src; a = f @ y / (f @ f)                     # 1차원 고정 특징 위의 선형 head
            res.append([r2(fit(X, y, np.zeros(p)), w_t, X_te), r2(fit(X, y, w_src), w_t, X_te), r2(a * w_src, w_t, X_te)])
        print(f"{'':12}{n:5d}      " + "        ".join(f"{v:7.3f}" for v in np.mean(res, 0)))

# %% cell 4 — 평가 자료의 중복: 정확한 일치로는 찾을 수 없다
from scipy import sparse

AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))                         # 아미노산 20종
corpus = rng.integers(20, size=(2000, 60))                           # 사전학습 서열 2000개, 길이 60
labels = rng.integers(2, size=2000)                                  # 서열마다의 가상 성질 (0/1)
near = corpus[:100].copy()                                           # 사전학습 서열 100개를 조금 바꾼 변이체
near[np.arange(100)[:, None], rng.integers(60, size=(100, 3))] = rng.integers(20, size=(100, 3))
test = np.vstack([near, rng.integers(20, size=(100, 60))])           # 시험 200개: 변이체 100 + 새 서열 100
y_test = np.r_[labels[:100], rng.integers(2, size=100)]              # 변이체는 원본의 성질을 물려받는다

def kmer_matrix(S, k=5):                                             # 서열 × k-mer 존재 여부 (희소 행렬)
    codes = sum(S[:, i:S.shape[1] - k + 1 + i] * 20 ** i for i in range(k))   # (n, 60-k+1)
    rows = np.repeat(np.arange(len(S)), codes.shape[1])
    M = sparse.csr_matrix((np.ones(codes.size), (rows, codes.ravel())), shape=(len(S), 20**k))
    M.data[:] = 1.0                                                  # 같은 k-mer가 여러 번 나와도 1
    return M

Kc, Kt = kmer_matrix(corpus), kmer_matrix(test)
inter = (Kt @ Kc.T).toarray()                                        # 공유 k-mer 수 (200, 2000)
jac = inter / (np.asarray(Kt.sum(1)) + np.asarray(Kc.sum(1)).T - inter)
corpus_str = {"".join(AA[c]) for c in corpus}
exact = sum("".join(AA[t]) in corpus_str for t in test)                # 문자열이 완전히 같은 서열
flag = jac.max(1) > 0.3
print(f"정확히 같은 서열: {exact}개    k-mer Jaccard > 0.3: {flag.sum()}개 (변이체 {flag[:100].sum()}, 새 서열 {flag[100:].sum()})")
pred = labels[jac.argmax(1)]                                         # "모델": 가장 닮은 사전학습 서열의 성질을 답한다
print(f"정확도  전체 {np.mean(pred == y_test):.2f}   변이체 {np.mean(pred[:100] == y_test[:100]):.2f}   새 서열 {np.mean(pred[100:] == y_test[100:]):.2f}")

# %% cell 5 — PyTorch: 다른 센서 영역으로 전이 — 처음부터·고정 특징·LoRA·full 미세조정
import copy

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(0)
t_s = torch.linspace(0, 1, 64)                                     # 시간 (T,), 단위 s
def domain(n, drift):                                              # 감쇠 진동 신호와 정규화한 ω. drift: 목표 영역의 센서 특성
    om, ga, ph = 2 * np.pi * (3 + 5 * torch.rand(n)), 0.5 + 3.5 * torch.rand(n), 2 * np.pi * torch.rand(n)
    x = torch.exp(-ga[:, None] * t_s) * torch.cos(om[:, None] * t_s + ph[:, None]) + 0.1 * torch.randn(n, 64)
    if drift:                                                      # 다른 센서: 기준선 drift, 이득 차이, 더 큰 잡음
        x = 0.6 * x + torch.randn(n, 1) * t_s + 0.2 * torch.randn(n, 64)
    return x, ((om - 2 * np.pi * 5.5) / (2 * np.pi * 1.5))[:, None]

class LoRA(nn.Module):                                             # y = W0 x + (α/r)·B A x, W0 고정
    def __init__(self, base, r=4, alpha=4):
        super().__init__(); self.base, self.s = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features**0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # B = 0: 시작점은 사전학습 모델 그대로
        base.requires_grad_(False)
    def forward(self, x):
        return self.base(x) + self.s * (x @ self.A.T) @ self.B.T

def mlp():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

def train(net, x, y, steps, lr=1e-3):
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        idx = torch.randint(len(x), (min(128, len(x)),))
        loss = ((net(x[idx]) - y[idx]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return net

pre = train(mlp(), *domain(20000, drift=False), steps=3000)        # 원천 영역에서 사전학습 (레이블 많음)
x_te, y_te = domain(2000, drift=True)
def r2(net):
    with torch.no_grad():
        return 1 - ((net(x_te) - y_te) ** 2).mean().item() / y_te.var().item()

def variant(kind):
    net = copy.deepcopy(pre)
    if kind == "고정 특징": net[:4].requires_grad_(False)        # 마지막 선형 head만 학습
    if kind == "LoRA": net[0], net[2] = LoRA(net[0]), LoRA(net[2])  # 은닉층은 저랭크 갱신만, head는 학습
    return mlp() if kind == "처음부터" else net

kinds = ["처음부터", "고정 특징", "LoRA", "full"]
n_train = [sum(p.numel() for p in variant(k).parameters() if p.requires_grad) for k in kinds]

print(f"사전학습 모델을 그대로 목표 영역에 쓰면 R² = {r2(pre):.3f}")
print("목표 레이블   처음부터   고정 특징   LoRA(r=4)   full 미세조정")
print("학습 파라미터 " + "  ".join(f"{k:>9,}" for k in n_train))
for n in [32, 128, 1024]:
    x, y = domain(n, drift=True)
    row = [r2(train(variant(k), x, y, steps=600)) for k in kinds]
    print(f"{n:8d}    " + "   ".join(f"{v:8.3f}" for v in row))
