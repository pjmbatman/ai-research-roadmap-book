"""18.8 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 5는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/architectures/representation-analysis/cells.py
"""
# %% cell 1 — linear probe와 대조: 무작위 신경망도 정보를 "담는다"
import numpy as np

rng = np.random.default_rng(0)
N, T = 2000, 64
t_s = np.linspace(0, 1, T)                              # 시간 (T,), 단위 s
omega = rng.uniform(2 * np.pi * 3, 2 * np.pi * 8, N)    # 각진동수 (N,), 단위 rad/s
gamma = rng.uniform(0.5, 4.0, N)                        # 감쇠율 (N,), 단위 1/s
phi = rng.uniform(0, 2 * np.pi, N)
X = np.exp(-gamma[:, None] * t_s) * np.cos(omega[:, None] * t_s + phi[:, None])
X += 0.1 * rng.normal(size=(N, T))                      # 감쇠 진동 신호 (N, T)
tr, te = np.arange(1500), np.arange(1500, N)

def probe_r2(Z, y, lam=1e-2):                           # 표준화 + ridge, test R²
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-8
    Zs = (Z - mu) / sd
    w = np.linalg.solve(Zs[tr].T @ Zs[tr] + lam * len(tr) * np.eye(Z.shape[1]), Zs[tr].T @ (y[tr] - y[tr].mean()))
    pred = Zs[te] @ w + y[tr].mean()
    return 1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()

W1 = rng.normal(size=(T, 256)) / np.sqrt(T)             # 학습하지 않은 무작위 층
W2 = rng.normal(size=(256, 256)) / np.sqrt(256)
H1 = np.maximum(0, X @ W1)                              # (N, 256)
H2 = np.maximum(0, H1 @ W2)                             # (N, 256)
print("표현                  ω probe R²   γ probe R²   ω 섞은 정답 R²")
for name, Z in [("입력 신호 (64)", X), ("무작위 층 1 (256)", H1), ("무작위 층 2 (256)", H2)]:
    print(f"{name:18}   {probe_r2(Z, omega):8.3f}   {probe_r2(Z, gamma):9.3f}   {probe_r2(Z, rng.permutation(omega)):10.3f}")

# %% cell 2 — linear CKA: 무엇에 불변이고 무엇에 속는가
def linear_cka(A, B):                                   # (N, D1), (N, D2) → 0과 1 사이
    A = A - A.mean(0); B = B - B.mean(0)                # 열 중심화
    return np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B))

n = 500                                                 # 계산량을 줄이려 표본 500개로 비교
A1, A2 = H1[:n], H2[:n]                                 # (n, 256)
Q, _ = np.linalg.qr(rng.normal(size=(256, 256)))        # 무작위 직교행렬
M = rng.normal(size=(256, 256))                         # 무작위 가역 행렬
print(f"CKA(H1, H1 Q)       = {linear_cka(A1, A1 @ Q):.3f}   (직교 변환)")
print(f"CKA(H1, 3 H1)       = {linear_cka(A1, 3 * A1):.3f}   (등방 스케일)")
print(f"CKA(H1, H1 M)       = {linear_cka(A1, A1 @ M):.3f}   (일반 가역 선형변환)")
print(f"CKA(H1, H2)         = {linear_cka(A1, A2):.3f}   (같은 망의 층 1, 2)")

W1b = rng.normal(size=(T, 256)) / np.sqrt(T)            # 다른 시드의 무작위 층
print(f"CKA(H1, 다른 망 H1) = {linear_cka(A1, np.maximum(0, X[:n] @ W1b)):.3f}   (학습 없는 두 망)")
noise_a, noise_b = rng.normal(size=(n, 256)), rng.normal(size=(n, 256))   # 서로 무관한 두 표현
big = 10 * rng.normal(size=(n, 1))                      # 두 표현이 공유하는 큰 분산 한 방향
print(f"CKA(무관, 무관)     = {linear_cka(noise_a, noise_b):.3f}")
print(f"CKA(무관+공통축, 무관+공통축) = {linear_cka(np.hstack([noise_a, big]), np.hstack([noise_b, big])):.3f}")

# %% cell 3 — 최근접 이웃 검색으로 표현 점검: 이웃의 물리량이 비슷한가
def nn_error(E, y):                                     # test 질의마다 train에서 최근접 이웃, |Δy| 평균
    d2 = (E[te] ** 2).sum(1)[:, None] - 2 * E[te] @ E[tr].T + (E[tr] ** 2).sum(1)[None]
    return np.abs(y[te] - y[tr][np.argmin(d2, axis=1)]).mean()

standardize = lambda Z: (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-8)
temp_mk = rng.normal(298_000, 2_000, size=(N, 1))      # 무관한 메타데이터: 센서 온도 (N, 1), 단위 mK
rand_nn = np.abs(omega[te] - omega[tr][rng.integers(len(tr), size=len(te))]).mean()
print(f"무작위 이웃                  |Δω| = {rand_nn:5.2f} rad/s")
for name, E in [("입력 신호", X), ("무작위 층 1", H1),
                ("층 1 + 센서 온도(mK)", np.hstack([H1, temp_mk])),
                ("위를 열마다 표준화", standardize(np.hstack([H1, temp_mk])))]:
    print(f"{name:20}  |Δω| = {nn_error(E, omega):5.2f} rad/s   |Δγ| = {nn_error(E, gamma):4.2f} 1/s")

# %% cell 4 — PCA 2차원 그림이 놓치는 것
fast = (omega > np.median(omega)).astype(float)         # 이진 레이블: 빠른 진동인가 (N,)
Zc = H1 - H1[tr].mean(0)
U, S, Vt = np.linalg.svd(Zc[tr], full_matrices=False)
evr = S**2 / (S**2).sum()                               # 설명 분산 비율

def probe_acc(Z):                                       # 최소제곱 선형 분류기, test 정확도
    Zb = np.hstack([Z, np.ones((N, 1))])
    w = np.linalg.lstsq(Zb[tr], 2 * fast[tr] - 1, rcond=None)[0]
    return ((Zb[te] @ w > 0) == fast[te]).mean()

for k in [2, 10, 50, 256]:
    print(f"PC {k:3d}개  설명 분산 {evr[:k].sum():.3f}   빠른 진동 판별 정확도 {probe_acc(Zc @ Vt[:k].T):.3f}")

# %% cell 5 — PyTorch: ω를 학습한 MLP의 층별 probe, 학습 전 같은 망과 비교
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
rng = np.random.default_rng(0)
N, T = 2000, 64
t_s = np.linspace(0, 1, T)                              # 앞 셀과 같은 감쇠 진동 자료
omega = rng.uniform(2 * np.pi * 3, 2 * np.pi * 8, N); gamma = rng.uniform(0.5, 4.0, N)
phi = rng.uniform(0, 2 * np.pi, N)
X = np.exp(-gamma[:, None] * t_s) * np.cos(omega[:, None] * t_s + phi[:, None]) + 0.1 * rng.normal(size=(N, T))
tr, te = np.arange(1500), np.arange(1500, N)

def probe_r2(Z, y, lam=1e-2):                           # 앞 셀과 같은 ridge probe
    Zs = (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-8)
    w = np.linalg.solve(Zs[tr].T @ Zs[tr] + lam * len(tr) * np.eye(Z.shape[1]), Zs[tr].T @ (y[tr] - y[tr].mean()))
    return 1 - ((y[te] - Zs[te] @ w - y[tr].mean()) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()

l1, l2 = nn.Sequential(nn.Linear(T, 128), nn.ReLU()), nn.Sequential(nn.Linear(128, 128), nn.ReLU())
net = nn.Sequential(l1, l2, nn.Linear(128, 1))         # (B, 64) → (B, 1): ω만 학습한다
xt = torch.tensor(X, dtype=torch.float32)               # (N, T)
yt = torch.tensor((omega - omega[tr].mean()) / omega[tr].std(), dtype=torch.float32)[:, None]

def layer_reps():                                       # 층 1, 2의 표현 (N, 128) 두 개
    with torch.no_grad():
        h1 = l1(xt); return [h1.numpy(), l2(h1).numpy()]

before = layer_reps()                                   # 학습 전(무작위 초기화) 같은 망
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for step in range(3000):
    idx = torch.as_tensor(rng.choice(tr, 128))
    loss = nn.functional.mse_loss(net(xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
after = layer_reps()
print("층    학습 전 ω R²  학습 후 ω R²   학습 전 γ R²  학습 후 γ R²")
for i in range(2):
    print(f"층 {i + 1}     {probe_r2(before[i], omega):6.3f}       {probe_r2(after[i], omega):6.3f}"
          f"        {probe_r2(before[i], gamma):6.3f}       {probe_r2(after[i], gamma):6.3f}")
