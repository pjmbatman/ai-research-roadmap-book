"""19.8 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 5는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/generative-and-foundation-models/self-supervised-learning/cells.py
"""
# %% cell 1 — InfoNCE(NT-Xent): 정렬·균일성·temperature
import numpy as np

rng = np.random.default_rng(0)
normalize = lambda z: z / np.linalg.norm(z, axis=1, keepdims=True)

def nt_xent(z1, z2, tau):                          # 두 view의 embedding (N, D), 2N-1개 중 짝 고르기
    z = normalize(np.vstack([z1, z2]))             # (2N, D), 단위 구 위로
    S = z @ z.T / tau                              # 코사인 유사도 / τ, (2N, 2N)
    np.fill_diagonal(S, -np.inf)                   # 자기 자신은 후보에서 뺀다
    N = len(z1)
    pos = np.r_[np.arange(N, 2 * N), np.arange(N)] # i의 짝: 다른 view의 같은 표본
    lse = np.log(np.exp(S - S.max(1, keepdims=True)).sum(1)) + S.max(1)
    return np.mean(lse - S[np.arange(2 * N), pos])

def align_uniform(z1, z2):                         # Wang·Isola의 두 성질 (작을수록 좋다)
    z1, z2 = normalize(z1), normalize(z2)
    z = np.vstack([z1, z2]); d2 = ((z[:, None] - z[None]) ** 2).sum(-1)
    iu = np.triu_indices(len(z), 1)
    return np.mean(((z1 - z2) ** 2).sum(1)), np.round(np.log(np.mean(np.exp(-2 * d2[iu]))), 3) + 0.0

N, D = 256, 16
base = rng.normal(size=(N, D))                     # 표본마다 고유한 방향
cases = {"좋은 표현": (base + 0.2 * rng.normal(size=(N, D)), base + 0.2 * rng.normal(size=(N, D))),
         "짝을 모름": (rng.normal(size=(N, D)), rng.normal(size=(N, D))),
         "완전 붕괴": (np.ones((N, D)) + 1e-3 * rng.normal(size=(N, D)), np.ones((N, D)) + 1e-3 * rng.normal(size=(N, D)))}
print(f"log(2N-1) = {np.log(2 * N - 1):.3f}")
print("표현          NT-Xent(τ=0.1)  NT-Xent(τ=0.5)   정렬     균일성")
for name, (z1, z2) in cases.items():
    a, u = align_uniform(z1, z2)
    print(f"{name:10}     {nt_xent(z1, z2, 0.1):7.3f}         {nt_xent(z1, z2, 0.5):7.3f}      {a:6.3f}   {u:6.3f}")

# %% cell 2 — 붕괴의 진단: 비대조 손실은 붕괴를 벌하지 않는다
def eff_rank(z):                                   # 특이값 분포의 엔트로피로 잰 유효 차원
    s = np.linalg.svd(z - z.mean(0), compute_uv=False)
    p = s / s.sum()
    return np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))

def vicreg_var(z, gamma=1.0):                      # 차원별 표준편차가 gamma 아래로 내려간 만큼의 벌점
    return np.mean(np.maximum(0, gamma - np.sqrt(z.var(0) + 1e-4)))

low = rng.normal(size=(N, 3)) @ rng.normal(size=(3, D))   # 3차원 부분공간에만 퍼진 표현
noise = lambda s: s * rng.normal(size=(N, D))
emb = {"건강한 표현": (base + noise(0.2), base + noise(0.2)),
       "차원 붕괴(3차원)": (low + noise(0.05), low + noise(0.05)),
       "완전 붕괴": (np.ones((N, D)) + noise(1e-3), np.ones((N, D)) + noise(1e-3))}
print("표현                정렬 MSE   유효 rank   분산 벌점   NT-Xent(τ=0.1)")
for name, (z1, z2) in emb.items():
    print(f"{name:14}     {np.mean((z1 - z2) ** 2):7.4f}    {eff_rank(z1):6.2f}     {vicreg_var(z1):6.3f}      {nt_xent(z1, z2, 0.1):6.3f}")

# %% cell 3 — shortcut: 손실이 낮은 표현이 좋은 표현은 아니다
from sklearn.linear_model import LogisticRegression

n = 1000
y = rng.integers(2, size=n)                        # 하위 과제의 레이블 (n,)
sem = (2 * y[:, None] - 1) * 0.8 + rng.normal(size=(n, 4))   # 의미 있는 특징: 레이블에 따라 이동 (n, 4)
tag = rng.normal(size=(n, 4))                      # 표본마다 고유한 표지(장비 ID, 배치 흔적 등) (n, 4)

def two_views(resample_tag):                       # 증강: 의미 특징에 잡음, 표지는 그대로 두거나 다시 뽑는다
    v = [np.hstack([sem + 0.7 * rng.normal(size=sem.shape),
                    rng.normal(size=tag.shape) if resample_tag else tag]) for _ in range(2)]
    return v                                        # 두 view, 각각 (n, 8)

read_tag, read_sem = (lambda v: v[:, 4:]), (lambda v: v[:, :4])   # 표지만 읽는 encoder, 의미만 읽는 encoder
probe = lambda Z: LogisticRegression().fit(Z[:500], y[:500]).score(Z[500:], y[500:])
print("증강              encoder       NT-Xent(τ=0.1)   선형 probe 정확도")
for aug, resample in [("표지 유지", False), ("표지 무작위화", True)]:
    v1, v2 = two_views(resample)
    for enc_name, enc in [("표지만 읽음", read_tag), ("의미만 읽음", read_sem)]:
        print(f"{aug:12}   {enc_name:10}     {nt_xent(enc(v1)[:256], enc(v2)[:256], 0.1):6.3f}           {probe(enc(v1)):.3f}")

# %% cell 4 — 사전학습의 효과는 레이블 수에 따라 달라진다
d, k = 100, 5
A = rng.normal(size=(k, d))                        # 잠재 요인 5개 → 관측 100차원
def make(n):                                       # 관측 x (n, d)와 레이블 y (n,)
    h = rng.normal(size=(n, k))
    return h @ A + 3.0 * rng.normal(size=(n, d)), (h[:, 0] + 0.5 * h[:, 1] > 0).astype(int)

X_unl, _ = make(5000)                              # 레이블 없는 사전학습 자료
_, _, Vt = np.linalg.svd(X_unl - X_unl.mean(0), full_matrices=False)
encoders = {"입력 그대로(100)": lambda X: X,
            "무작위 투영(5)": lambda X, R=rng.normal(size=(d, k)): X @ R,
            "사전학습 PCA(5)": lambda X: (X - X_unl.mean(0)) @ Vt[:k].T}   # 선형 autoencoder = PCA
X_te, y_te = make(2000)
print("레이블 수   " + "   ".join(f"{name:>14}" for name in encoders))
for n_lab in [10, 30, 100, 1000]:
    accs = {name: [] for name in encoders}
    for rep in range(10):                          # 레이블 표본을 바꿔 10번 반복
        X_tr, y_tr = make(n_lab)
        for name, enc in encoders.items():         # 모두 같은 probe, 같은 정규화 강도
            accs[name].append(LogisticRegression(C=1.0, max_iter=1000).fit(enc(X_tr), y_tr).score(enc(X_te), y_te))
    print(f"{n_lab:6d}     " + "   ".join(f"{np.mean(a):14.3f}" for a in accs.values()))

# %% cell 5 — PyTorch: 감쇠 진동 신호의 대조 학습, 무작위 초기화·붕괴와 비교
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
torch.manual_seed(0)
rng = np.random.default_rng(0)
t_s = torch.linspace(0, 1, 64)                                     # 시간 (T,), 단위 s
def signals(omega, gamma):                                         # 같은 (ω, γ)의 새 view: 위상·진폭·잡음은 무작위
    B = len(omega)
    phi, amp = torch.rand(B, 1) * 2 * np.pi, 0.5 + torch.rand(B, 1)
    return amp * torch.exp(-gamma[:, None] * t_s) * torch.cos(omega[:, None] * t_s + phi) + 0.1 * torch.randn(B, 64)
draw = lambda B: (2 * np.pi * (3 + 5 * torch.rand(B)), 0.5 + 3.5 * torch.rand(B))   # ω (rad/s), γ (1/s)

def nt_xent(z1, z2, tau=0.2):                                      # (B, D) 두 개 → 스칼라
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    S = z @ z.T / tau - 1e9 * torch.eye(len(z))                    # 자기 자신 제외
    B = len(z1)
    return F.cross_entropy(S, torch.cat([torch.arange(B, 2 * B), torch.arange(B)]))

def encoder():                                                     # backbone(평가 대상)과 projection head
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()), nn.Linear(128, 64)

def train(loss_name, steps=1500):
    f, g = encoder(); opt = torch.optim.Adam([*f.parameters(), *g.parameters()], lr=1e-3)
    for _ in range(steps):
        om, ga = draw(256)
        z1, z2 = g(f(signals(om, ga))), g(f(signals(om, ga)))      # 같은 계열의 두 view (256, 64)
        loss = nt_xent(z1, z2) if loss_name == "contrastive" else ((z1 - z2) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return f

def eff_rank(z):                                                   # cell 2와 같은 유효 rank
    s = torch.linalg.svdvals(z - z.mean(0)); p = s / s.sum()
    return torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()

def probe_r2(Z, y, n_tr):                                          # ridge probe, 앞 n_tr개로 학습, 마지막 1000개로 시험
    Z = (Z - Z[:n_tr].mean(0)) / (Z[:n_tr].std(0) + 1e-6)
    w = torch.linalg.solve(Z[:n_tr].T @ Z[:n_tr] + 1e-2 * n_tr * torch.eye(Z.shape[1]), Z[:n_tr].T @ (y[:n_tr] - y[:n_tr].mean()))
    pred = Z[-1000:] @ w + y[:n_tr].mean()
    return (1 - ((y[-1000:] - pred) ** 2).mean() / y[-1000:].var()).item()

om, ga = draw(3000); X = signals(om, ga)                           # 평가용 자료 (3000, 64)
models = {"입력 그대로": None, "무작위 초기화": encoder()[0], "대조 학습": train("contrastive"), "정렬만(비대조)": train("align")}
print("표현              평균 std   유효 rank   ω probe R² (레이블 100)   (레이블 1000)   γ probe R² (1000)")
with torch.no_grad():
    for name, enc in models.items():
        Z = X if enc is None else enc(X)
        print(f"{name:12}     {Z.std(0).mean().item():7.3f}   {eff_rank(Z):7.2f}        {probe_r2(Z, om, 100):7.3f}              {probe_r2(Z, om, 1000):7.3f}        {probe_r2(Z, ga, 1000):7.3f}")
