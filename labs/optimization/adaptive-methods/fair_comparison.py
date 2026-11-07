"""10.4 실습: optimizer만 바꾼 비교와 학습률까지 튜닝한 비교에서 결론이 어떻게 달라지는지 본다.

문제: 1차원 회귀 y = sin(2x) + 잡음을 은닉 32개(tanh)짜리 MLP로 맞춘다. NumPy로 직접 역전파한다.
optimizer: SGD(momentum 0.9), Adam(beta1 0.9, beta2 0.999)
- 비교 A: 두 optimizer 모두 학습률 1e-3 (흔히 쓰는 Adam 기본값)
- 비교 B: optimizer마다 학습률 격자(1e-4 ~ 1, 반 decade 간격)를 탐색하고 검증 손실이 가장 낮은 학습률을 고른다
각 설정은 시드 3개(초기화와 mini-batch 순서)의 평균 검증 MSE로 고르고, 고른 설정의 시험(test) MSE를 따로 보고한다.

실행: uv run --project labs python labs/optimization/adaptive-methods/fair_comparison.py
"""
import numpy as np

STEPS, BATCH, HIDDEN = 3000, 32, 32
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
SEEDS = [0, 1, 2]


def make_data(rng, n):
    x = rng.uniform(-3, 3, size=(n, 1))
    return x, np.sin(2 * x) + 0.1 * rng.normal(size=(n, 1))


def init(rng):
    return {"W1": rng.normal(size=(1, HIDDEN)), "b1": np.zeros(HIDDEN),
            "W2": rng.normal(size=(HIDDEN, 1)) / np.sqrt(HIDDEN), "b2": np.zeros(1)}


def forward(p, x):
    h = np.tanh(x @ p["W1"] + p["b1"])                 # (B, H)
    return h, h @ p["W2"] + p["b2"]                    # (B, 1)


def grads(p, x, y):
    h, out = forward(p, x)
    d_out = 2 * (out - y) / len(y)                     # dMSE/dout, (B, 1)
    d_h = d_out @ p["W2"].T * (1 - h**2)               # (B, H)
    return {"W1": x.T @ d_h, "b1": d_h.sum(0), "W2": h.T @ d_out, "b2": d_out.sum(0)}


def train(opt, lr, seed, data):
    xtr, ytr, xva, yva, xte, yte = data
    rng = np.random.default_rng(seed)
    p = init(rng)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v_) for k, v_ in p.items()}
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(1, STEPS + 1):
            idx = rng.choice(len(xtr), size=BATCH, replace=False)
            g = grads(p, xtr[idx], ytr[idx])
            for k in p:
                if opt == "sgdm":
                    m[k] = 0.9 * m[k] + g[k]
                    p[k] = p[k] - lr * m[k]
                else:
                    m[k] = 0.9 * m[k] + 0.1 * g[k]
                    v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                    p[k] = p[k] - lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(v[k] / (1 - 0.999**t)) + 1e-8)
        val = float(np.mean((forward(p, xva)[1] - yva) ** 2))
        test = float(np.mean((forward(p, xte)[1] - yte) ** 2))
    return (val, test) if np.isfinite(val) else (np.inf, np.inf)


def main():
    rng = np.random.default_rng(123)
    data = (*make_data(rng, 512), *make_data(rng, 512), *make_data(rng, 2000))   # 학습, 검증, 시험
    names = {"sgdm": "SGD+momentum", "adam": "Adam"}
    runs = {o: np.array([[train(o, lr, s, data) for s in SEEDS] for lr in LRS]) for o in names}   # (lr, seed, 2)
    table = {o: runs[o][:, :, 0].mean(axis=1) for o in names}                                   # 검증 MSE 평균
    test = {o: runs[o][:, :, 1] for o in names}

    print(f"검증 MSE (시드 {len(SEEDS)}개 평균, {STEPS} step, batch {BATCH}). 잡음 분산 = 0.01")
    print("lr        " + "".join(f"{lr:>9.0e}" for lr in LRS))
    for o, name in names.items():
        print(f"{name:12s}" + "".join(f"{v:9.3f}" if np.isfinite(v) else f"{'발산':>8s}" for v in table[o]))

    i = LRS.index(1e-3)
    print("\n비교 A: 같은 학습률 1e-3")
    for o, name in names.items():
        print(f"  {name:12s} 시험 MSE {test[o][i].mean():.3f} (시드별 {np.round(test[o][i], 3)})")
    print("비교 B: optimizer마다 학습률을 튜닝")
    for o, name in names.items():
        j = int(np.argmin(table[o]))
        edge = " (격자 끝: 범위를 넓혀야 함)" if j in (0, len(LRS) - 1) else ""
        print(f"  {name:12s} 검증으로 고른 lr {LRS[j]:.0e}  시험 MSE {test[o][j].mean():.3f} (시드별 {np.round(test[o][j], 3)}){edge}")


if __name__ == "__main__":
    main()
