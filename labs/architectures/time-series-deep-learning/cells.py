"""18.7 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 6은 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/architectures/time-series-deep-learning/cells.py
"""
# %% cell 1 — patching: window를 겹치는 조각으로 잘라 토큰으로 쓰기
import numpy as np

def patchify(x, P, S):                                  # (B, C, L) → (B*C, n_patch, P)
    B, C, L_in = x.shape
    n_patch = (L_in - P) // S + 1
    idx = S * np.arange(n_patch)[:, None] + np.arange(P)[None, :]   # (n_patch, P)
    return x.reshape(B * C, L_in)[:, idx]               # 채널마다 따로: channel independence

x = np.random.default_rng(1).normal(size=(32, 3, 96))  # batch 32, 채널 3, lookback 96
tokens = patchify(x, P=16, S=8)
print("입력", x.shape, "→ 패치 토큰", tokens.shape)
print("첫 두 패치가 원래 시점 0–15, 8–23인가:",
      np.array_equal(tokens[0, 0], x[0, 0, 0:16]), np.array_equal(tokens[0, 1], x[0, 0, 8:24]))
n_tok = tokens.shape[1]
print(f"토큰 수 96 → {n_tok},  attention 행렬 원소 수 비 {96**2 / n_tok**2:.1f}배 감소")

# %% cell 2 — 시계열을 (입력 window, 예측 horizon) 쌍으로 자르기
import numpy as np

rng = np.random.default_rng(0)
n_h = 24 * 120                                          # 시간 단위 관측 120일
t_h = np.arange(n_h)
noise = np.zeros(n_h)
for i in range(1, n_h):                                 # AR(1) 잡음: 날씨처럼 며칠 이어지는 편차
    noise[i] = 0.97 * noise[i - 1] + 0.4 * rng.normal()
temp_c = 10 + 0.003 * t_h + 5 * np.sin(2 * np.pi * (t_h - 9) / 24) + noise   # (n_h,)

L, H = 48, 24                                           # lookback 48시간, horizon 24시간
def make_windows(x, L, H, stride=1):
    starts = np.arange(0, len(x) - L - H + 1, stride)
    X = np.stack([x[s:s + L] for s in starts])          # (n_win, L)
    Y = np.stack([x[s + L:s + L + H] for s in starts])  # (n_win, H)
    return X, Y, starts

X, Y, starts = make_windows(temp_c, L, H)
print("X", X.shape, " Y", Y.shape)

idx = rng.permutation(len(X)); n_tr = int(0.8 * len(X))
tr, te = idx[:n_tr], idx[n_tr:]                         # 무작위 분할 (잘못된 예)
covered = np.zeros(n_h, bool)
for s in starts[tr]:
    covered[s:s + L + H] = True                         # train window가 본 시점
frac = np.mean([covered[s + L:s + L + H].all() for s in starts[te]])
print(f"무작위 분할: test 정답 구간을 train window가 이미 본 비율 = {frac:.3f}")

# %% cell 3 — 시간 순서 분할과 나이브 baseline, MASE
cut = 24 * 96                                           # 앞 96일 학습, 뒤 24일 평가
is_tr = starts + L + H <= cut                           # 정답까지 cut 이전에 끝나는 window
is_te = starts + L >= cut                               # 정답이 모두 cut 이후인 window
Xtr, Ytr, Xte, Yte = X[is_tr], Y[is_tr], X[is_te], Y[is_te]

baselines = {
    "train 평균     ": np.full_like(Yte, temp_c[:cut].mean()),
    "persistence    ": np.repeat(Xte[:, -1:], H, axis=1),   # 마지막 관측값 유지
    "seasonal naive ": Xte[:, -24:],                         # 24시간 전 값 (H = 24)
}
scale = np.abs(temp_c[24:cut] - temp_c[:cut - 24]).mean()  # 학습 구간 seasonal naive 1-step MAE
print("모델              h=1    h=6   h=24   평균 MAE   MASE")
for name, pred in baselines.items():
    mae_h = np.abs(pred - Yte).mean(0)                   # horizon별 MAE (H,)
    print(f"{name}  {mae_h[0]:5.2f}  {mae_h[5]:5.2f}  {mae_h[23]:5.2f}   {mae_h.mean():6.2f}   {mae_h.mean() / scale:5.2f}")

# %% cell 4 — 같은 모델, 두 분할: 무작위 분할의 낙관 편향
def fit_linear(X, Y, lam=1.0):                          # 다중 출력 ridge (direct 전략)
    Xb = np.hstack([X, np.ones((len(X), 1))])           # (n, L+1)
    W = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)   # (L+1, H)
    return lambda Xn: np.hstack([Xn, np.ones((len(Xn), 1))]) @ W

def fit_1nn(X, Y):                                      # 가장 비슷한 과거 window의 미래를 복사
    def predict(Xn):
        d2 = (Xn**2).sum(1)[:, None] - 2 * Xn @ X.T + (X**2).sum(1)[None]
        return Y[np.argmin(d2, axis=1)]
    return predict

splits = {"무작위 분할  ": (X[tr], Y[tr], X[te], Y[te]), "시간 순서 분할": (Xtr, Ytr, Xte, Yte)}
for model_name, fit in [("선형(ridge)   ", fit_linear), ("1-최근접 이웃 ", fit_1nn)]:
    for split_name, (a, b, c, d) in splits.items():
        print(f"{model_name} {split_name}  test MAE {np.abs(fit(a, b)(c) - d).mean():.2f}")
print(f"seasonal naive 시간 순서 분할  test MAE {np.abs(Xte[:, -24:] - Yte).mean():.2f}")

# %% cell 5 — 다단계 예측: recursive와 direct
def recursive(Xtr, Ytr, Xte):
    one_step = fit_linear(Xtr, Ytr[:, :1])              # 1시간 뒤만 예측하는 모델
    window, out = Xte.copy(), []                        # (n_te, L')
    for h in range(H):                                  # 예측값을 입력 끝에 붙여 24번 반복
        y_next = one_step(window)                       # (n_te, 1)
        out.append(y_next[:, 0])
        window = np.hstack([window[:, 1:], y_next])
    return np.stack(out, axis=1)                        # (n_te, H)

print("lookback  전략         h=1    h=6   h=12   h=24")
for Ls in [48, 6]:                                      # 하루 주기를 보는 입력 vs 못 보는 입력
    a, c = Xtr[:, -Ls:], Xte[:, -Ls:]
    for name, pred in [("recursive", recursive(a, Ytr, c)), ("direct   ", fit_linear(a, Ytr)(c))]:
        m = np.abs(pred - Yte).mean(0)
        print(f"{Ls:4d}시간  {name}   {m[0]:5.2f}  {m[5]:5.2f}  {m[11]:5.2f}  {m[23]:5.2f}")

# %% cell 6 — PyTorch: 같은 분할·같은 metric으로 선형층과 MLP 비교
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
rng = np.random.default_rng(0)
n_h = 24 * 120; t_h = np.arange(n_h); noise = np.zeros(n_h)
for i in range(1, n_h):                                 # 본문 앞 셀과 같은 자료
    noise[i] = 0.97 * noise[i - 1] + 0.4 * rng.normal()
temp_c = 10 + 0.003 * t_h + 5 * np.sin(2 * np.pi * (t_h - 9) / 24) + noise
L, H, cut = 48, 24, 24 * 96
s = np.arange(n_h - L - H + 1)
X = np.stack([temp_c[i:i + L] for i in s]); Y = np.stack([temp_c[i + L:i + L + H] for i in s])
tr, te = s + L + H <= cut, s + L >= cut                 # 앞과 같은 시간 순서 분할
mu, sd = temp_c[:cut].mean(), temp_c[:cut].std()        # 정규화 통계는 학습 구간에서만
f = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
Xtr, Ytr, Xte = f(X[tr]), f(Y[tr]), f(X[te])            # (n_tr, L), (n_tr, H), (n_te, L)

def train(model, epochs=60):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xtr)).split(64):
            loss = nn.functional.mse_loss(model(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return model(Xte).numpy() * sd + mu             # 원래 단위(°C)로 되돌린다 (n_te, H)

models = {"선형 1층          ": nn.Linear(L, H),
          "MLP 2층 (은닉 256)": nn.Sequential(nn.Linear(L, 256), nn.ReLU(), nn.Linear(256, H))}
for name, m in models.items():
    print(f"{name}  test MAE {np.abs(train(m) - Y[te]).mean():.3f} °C")
print(f"seasonal naive      test MAE {np.abs(X[te][:, -24:] - Y[te]).mean():.3f} °C")
