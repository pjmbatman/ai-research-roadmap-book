"""1.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-landscape/first-pipeline/cells.py
같은 파이프라인을 스크립트 하나로 실행하려면: uv run --project labs python labs/research-landscape/first-pipeline/pipeline.py
"""
# %% cell 1 — 문제 정의와 데이터
import numpy as np
import pandas as pd
from scipy.special import ellipk

G_MS2 = 9.81                                               # 중력가속도 [m/s^2]

def make_pendulum_data(n, seed):
    """진자 n개의 주기 측정을 흉내 낸다. 참 주기는 타원적분 공식, 측정 잡음은 0.01 s."""
    r = np.random.default_rng(seed)
    length_m = r.uniform(0.2, 2.0, size=n)                 # (n,) 줄 길이
    amp_deg = r.uniform(5.0, 80.0, size=n)                 # (n,) 처음 진폭
    m = np.sin(np.deg2rad(amp_deg) / 2) ** 2
    period_s = 4 * np.sqrt(length_m / G_MS2) * ellipk(m)   # (n,) 참 주기
    period_s = period_s + r.normal(0, 0.01, size=n)        # 스톱워치 측정 잡음
    return pd.DataFrame({"length_m": length_m, "amp_deg": amp_deg, "period_s": period_s})

df = make_pendulum_data(n=300, seed=0)
print(df.shape)
print(df.head(3).round(3))

# %% cell 2 — 분할과 baseline
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(1)
idx = rng.permutation(len(df))                             # 한 번 섞고, 이후 고정
tr, va, te = idx[:180], idx[180:240], idx[240:]            # 60% / 20% / 20%
X = df[["length_m", "amp_deg"]].to_numpy()                 # (300, 2)
y = df["period_s"].to_numpy()                              # (300,)

def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))

def physics_features(X_):                                  # √L, √L·θ², √L·θ⁴ (θ: 라디안)
    sq, th = np.sqrt(X_[:, 0]), np.deg2rad(X_[:, 1])
    return np.column_stack([sq, sq * th**2, sq * th**4])

lin = LinearRegression().fit(X[tr], y[tr])
phys = LinearRegression().fit(physics_features(X[tr]), y[tr])
baselines = {
    "평균 예측": lambda X_: np.full(len(X_), y[tr].mean()),
    "선형 회귀 (L, 진폭)": lin.predict,
    "작은 진폭 공식 2π√(L/g)": lambda X_: 2 * np.pi * np.sqrt(X_[:, 0] / G_MS2),
    "물리 특징 + 선형 회귀": lambda X_: phys.predict(physics_features(X_)),
}
for name, predict in baselines.items():
    print(f"검증 RMSE {rmse(predict(X[va]), y[va]):.4f} s   {name}")

# %% cell 3 — 작은 MLP: forward → loss → backward → update
mu_x, sd_x = X[tr].mean(0), X[tr].std(0)                   # 표준화 통계는 학습 데이터에서만
mu_y, sd_y = y[tr].mean(), y[tr].std()

def train_mlp(seed, hidden=32, steps=3000, lr=0.05, verbose=False):
    r = np.random.default_rng(seed)
    W1, b1 = r.normal(0, 1 / np.sqrt(2), (2, hidden)), np.zeros(hidden)          # (2, H), (H,)
    W2, b2 = r.normal(0, 1 / np.sqrt(hidden), (hidden, 1)), np.zeros(1)          # (H, 1), (1,)
    Xs, ys = (X[tr] - mu_x) / sd_x, ((y[tr] - mu_y) / sd_y)[:, None]             # (N, 2), (N, 1)
    for step in range(steps):
        h = np.tanh(Xs @ W1 + b1)                          # forward  (N, H)
        out = h @ W2 + b2                                  #          (N, 1)
        loss = np.mean((out - ys) ** 2)                    # loss     MSE
        d_out = 2 * (out - ys) / len(ys)                   # backward (N, 1)
        dW2, db2 = h.T @ d_out, d_out.sum(0)
        d_h = (d_out @ W2.T) * (1 - h ** 2)                # (N, H)
        dW1, db1 = Xs.T @ d_h, d_h.sum(0)
        W1, b1, W2, b2 = W1 - lr * dW1, b1 - lr * db1, W2 - lr * dW2, b2 - lr * db2   # update
        if verbose and step in (0, 100, 1000, steps - 1):
            print(f"step {step:4d}  학습 loss {loss:.4f}")
    return lambda X_: (np.tanh((X_ - mu_x) / sd_x @ W1 + b1) @ W2 + b2)[:, 0] * sd_y + mu_y

mlp = train_mlp(seed=0, verbose=True)
print(f"검증 RMSE {rmse(mlp(X[va]), y[va]):.4f} s   MLP")

# %% cell 4 — 테스트 평가와 보고
models = {**baselines, "MLP (은닉 32)": mlp}
print("RMSE [s]   학습    검증    테스트   모델")
for name, predict in models.items():
    scores = [rmse(predict(X[s]), y[s]) for s in (tr, va, te)]
    print("         " + "  ".join(f"{v:.4f}" for v in scores) + f"   {name}")
print("측정 잡음의 표준편차 0.0100")

# %% cell 5 — 다시 실행하면 같은 숫자가 나오는가
for seed in range(5):
    print(f"초기화 seed {seed}  MLP 검증 RMSE {rmse(train_mlp(seed)(X[va]), y[va]):.4f} s")
