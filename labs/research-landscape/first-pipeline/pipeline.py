"""1.5 첫 번째 연구 파이프라인: 진자 주기 회귀를 스크립트 하나로 완주한다.

문제 정의 → 데이터 → 분할 → baseline → 신경망 → 평가 → 보고.
본문 코드 셀(cells.py)과 같은 계산을 한 프로세스에서 순서대로 실행하고, 결과 표를 출력한다.

실행 (저장소 루트에서):
    uv run --project labs python labs/research-landscape/first-pipeline/pipeline.py
    uv run --project labs python labs/research-landscape/first-pipeline/pipeline.py --split-seed 2 --init-seeds 0 1 2 3 4
"""
import argparse
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import ellipk
from sklearn.linear_model import LinearRegression

G_MS2 = 9.81                                               # 중력가속도 [m/s^2]
NOISE_S = 0.01                                             # 주기 측정 잡음의 표준편차 [s]


def make_pendulum_data(n, seed):
    """진자 n개의 주기 측정을 흉내 낸다. 참 주기는 타원적분 공식, 측정 잡음은 NOISE_S."""
    r = np.random.default_rng(seed)
    length_m = r.uniform(0.2, 2.0, size=n)                 # (n,)
    amp_deg = r.uniform(5.0, 80.0, size=n)                 # (n,)
    m = np.sin(np.deg2rad(amp_deg) / 2) ** 2
    period_s = 4 * np.sqrt(length_m / G_MS2) * ellipk(m)   # (n,)
    period_s = period_s + r.normal(0, NOISE_S, size=n)
    return pd.DataFrame({"length_m": length_m, "amp_deg": amp_deg, "period_s": period_s})


def split_indices(n, seed):
    idx = np.random.default_rng(seed).permutation(n)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def physics_features(X):
    """작은 진폭 전개 T ≈ 2π√(L/g)(1 + θ²/16 + 11θ⁴/3072)에서 온 특징: √L, √L·θ², √L·θ⁴."""
    sq, th = np.sqrt(X[:, 0]), np.deg2rad(X[:, 1])
    return np.column_stack([sq, sq * th**2, sq * th**4])   # (N, 3)


def train_mlp(X, y, seed, hidden=32, steps=3000, lr=0.05):
    """은닉층 하나(tanh)의 MLP를 전체 batch 경사하강으로 학습하고 예측 함수를 반환한다."""
    mu_x, sd_x = X.mean(0), X.std(0)                       # 학습 데이터의 통계만 쓴다
    mu_y, sd_y = y.mean(), y.std()
    r = np.random.default_rng(seed)
    W1, b1 = r.normal(0, 1 / np.sqrt(X.shape[1]), (X.shape[1], hidden)), np.zeros(hidden)
    W2, b2 = r.normal(0, 1 / np.sqrt(hidden), (hidden, 1)), np.zeros(1)
    Xs, ys = (X - mu_x) / sd_x, ((y - mu_y) / sd_y)[:, None]
    for _ in range(steps):
        h = np.tanh(Xs @ W1 + b1)                          # (N, H)
        out = h @ W2 + b2                                  # (N, 1)
        d_out = 2 * (out - ys) / len(ys)
        dW2, db2 = h.T @ d_out, d_out.sum(0)
        d_h = (d_out @ W2.T) * (1 - h**2)
        dW1, db1 = Xs.T @ d_h, d_h.sum(0)
        W1, b1, W2, b2 = W1 - lr * dW1, b1 - lr * db1, W2 - lr * dW2, b2 - lr * db2
    return lambda X_: (np.tanh((X_ - mu_x) / sd_x @ W1 + b1) @ W2 + b2)[:, 0] * sd_y + mu_y


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=300, help="측정 수")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[0], help="MLP 초기화 seed들")
    args = parser.parse_args()

    df = make_pendulum_data(args.n, args.data_seed)
    X, y = df[["length_m", "amp_deg"]].to_numpy(), df["period_s"].to_numpy()
    tr, va, te = split_indices(len(df), args.split_seed)

    lin = LinearRegression().fit(X[tr], y[tr])
    phys = LinearRegression().fit(physics_features(X[tr]), y[tr])
    models = {
        "평균 예측": lambda X_: np.full(len(X_), y[tr].mean()),
        "선형 회귀 (L, 진폭)": lin.predict,
        "작은 진폭 공식 2π√(L/g)": lambda X_: 2 * np.pi * np.sqrt(X_[:, 0] / G_MS2),
        "물리 특징 + 선형 회귀": lambda X_: phys.predict(physics_features(X_)),
    }
    for s in args.init_seeds:
        models[f"MLP (은닉 32, seed {s})"] = train_mlp(X[tr], y[tr], seed=s)

    print(f"# 진자 주기 회귀 — n={args.n}, data_seed={args.data_seed}, split_seed={args.split_seed}")
    print(f"# python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, "
          f"scikit-learn {sklearn.__version__}")
    print(f"# 분할: 학습 {len(tr)} / 검증 {len(va)} / 테스트 {len(te)}, 측정 잡음 {NOISE_S} s")
    print("RMSE [s]   학습    검증    테스트   모델")
    for name, predict in models.items():
        scores = [rmse(predict(X[s]), y[s]) for s in (tr, va, te)]
        print("         " + "  ".join(f"{v:.4f}" for v in scores) + f"   {name}")


if __name__ == "__main__":
    main()
