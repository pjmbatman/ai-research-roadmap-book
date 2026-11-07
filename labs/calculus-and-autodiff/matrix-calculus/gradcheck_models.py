"""8.2 실습: 선형회귀와 2층 MLP의 gradient를 손으로 유도해 구현하고 수치미분으로 검증한다.

1. 여러 크기·시드에서 손 유도 gradient와 중앙차분 gradient의 상대오차를 잰다.
2. 일부러 틀린 구현 두 가지(전치 누락, ReLU 마스크 누락)가 gradient check에서 어떻게 드러나는지 본다.
3. 손 유도 gradient만으로 2층 MLP를 학습해 손실이 줄어드는지 확인한다.

실행: uv run --project labs python labs/calculus-and-autodiff/matrix-calculus/gradcheck_models.py
"""
import numpy as np


def num_grad(f, X, eps=1e-6):
    G = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        E = np.zeros_like(X)
        E[idx] = eps
        G[idx] = (f(X + E) - f(X - E)) / (2 * eps)
    return G


def rel_err(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12)


# ---------------------------------------------------------------- 선형회귀
def linreg_loss(w, b, X, y):
    r = X @ w + b - y                     # (N,)
    return 0.5 * np.mean(r**2)


def linreg_grads(w, b, X, y):
    r = X @ w + b - y                     # (N,)
    return X.T @ r / len(y), r.mean()     # (D,), ()


# ---------------------------------------------------------------- 2층 MLP
def mlp_forward(P, X, Y):
    A1 = X @ P["W1"].T + P["b1"]          # (N, H)
    Hh = np.maximum(A1, 0)                # (N, H)
    Z = Hh @ P["W2"].T + P["b2"]          # (N, K)
    Z = Z - Z.max(axis=1, keepdims=True)
    logp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))   # (N, K)
    return A1, Hh, np.exp(logp), -np.mean(np.sum(Y * logp, axis=1))


def mlp_grads(P, X, Y, bug=None):
    A1, Hh, Pr, _ = mlp_forward(P, X, Y)
    Gz = (Pr - Y) / len(X)                # (N, K)
    if bug == "no-transpose":             # W2^T 대신 W2를 쓴 것처럼: 정사각일 때만 shape이 맞는다
        Ga = Gz @ P["W2"].T
    else:
        Ga = Gz @ P["W2"]                 # (N, H)
    if bug != "no-mask":
        Ga = Ga * (A1 > 0)
    return {"W1": Ga.T @ X, "b1": Ga.sum(0), "W2": Gz.T @ Hh, "b2": Gz.sum(0)}


def init(D, H, K, rng, scale=1.0):
    return {"W1": rng.normal(size=(H, D)) * scale / np.sqrt(D), "b1": np.zeros(H),
            "W2": rng.normal(size=(K, H)) * scale / np.sqrt(H), "b2": np.zeros(K)}


def check_linreg():
    print("[1] 선형회귀: 손 유도 vs 중앙차분 (상대오차)")
    for seed, (N, D) in enumerate([(20, 2), (100, 5), (500, 10)]):
        rng = np.random.default_rng(seed)
        X, y = rng.normal(size=(N, D)), rng.normal(size=N)
        w, b = rng.normal(size=D), 0.3
        gw, gb = linreg_grads(w, b, X, y)
        nw = num_grad(lambda v: linreg_loss(v, b, X, y), w)
        nb = num_grad(lambda v: linreg_loss(w, v[0], X, y), np.array([b]))[0]
        print(f"  N={N:4d} D={D:3d}   w {rel_err(gw, nw):.1e}   b {rel_err(np.array([gb]), np.array([nb])):.1e}")


def check_mlp():
    print("[2] 2층 MLP: 손 유도 vs 중앙차분 (파라미터별 상대오차)")
    for seed, (N, D, H, K) in enumerate([(8, 3, 5, 4), (16, 4, 4, 4), (32, 6, 10, 3)]):
        rng = np.random.default_rng(seed)
        X, Y = rng.normal(size=(N, D)), np.eye(K)[rng.integers(0, K, size=N)]
        P = init(D, H, K, rng)
        for bug in (None, "no-mask", "no-transpose"):
            if bug == "no-transpose" and H != K:
                continue                  # shape이 맞지 않아 바로 오류가 나는 경우는 건너뛴다
            g = mlp_grads(P, X, Y, bug)
            errs = []
            for k in ("W1", "b1", "W2", "b2"):
                n = num_grad(lambda v, k=k: mlp_forward({**P, k: v}, X, Y)[3], P[k])
                errs.append(f"{k} {rel_err(g[k], n):.1e}")
            print(f"  N={N:2d} D={D} H={H:2d} K={K} {str(bug or '정상'):13}  " + "  ".join(errs))


def train_mlp():
    print("[3] 손 유도 gradient로 2층 MLP 학습 (3개 클래스, 2차원 입력)")
    rng = np.random.default_rng(0)
    N, K = 300, 3
    labels = np.repeat(np.arange(K), N // K)
    t = rng.uniform(0, 1, size=N)
    angle = labels * 2 * np.pi / K + 3.0 * t + 0.2 * rng.normal(size=N)
    X = np.stack([t * np.cos(angle), t * np.sin(angle)], axis=1)   # (N, 2) 나선형
    Y = np.eye(K)[labels]
    P = init(2, 32, K, rng, scale=2.0)
    lr = 1.0
    for step in range(2001):
        if step % 500 == 0:
            _, _, Pr, loss = mlp_forward(P, X, Y)
            acc = np.mean(Pr.argmax(1) == labels)
            print(f"  step {step:4d}  loss {loss:.4f}  정확도 {acc:.3f}")
        g = mlp_grads(P, X, Y)
        P = {k: P[k] - lr * g[k] for k in P}


if __name__ == "__main__":
    check_linreg()
    check_mlp()
    train_mlp()
