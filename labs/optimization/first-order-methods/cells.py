"""10.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/optimization/first-order-methods/cells.py
"""
# %% cell 1 — 학습률과 곡률: 최대 안정 학습률 2/L, 조건수와 반복 수
import numpy as np

lam = np.array([1.0, 10.0])                  # Hessian 고윳값: mu = 1, L = 10
for lr in [0.05, 0.10, 0.15, 0.19, 0.21]:    # 2 / L = 0.2
    x = np.ones(2)
    for _ in range(100):
        x = x - lr * lam * x                 # grad = H x, H = diag(lam)
    rates = np.abs(1 - lr * lam)             # 좌표별 축소율 |1 - lr * lambda|
    print(f"lr = {lr:.2f}  축소율 {rates.round(2)}  100 step 후 |x| = {np.linalg.norm(x):.2e}")

for kappa in [10, 100, 1000]:
    lam_k, x, k = np.array([1.0, kappa]), np.ones(2), 0
    lr = 2 / (1 + kappa)                     # 이차함수에서 최적인 고정 학습률
    while np.linalg.norm(x) >= 1e-6:
        x, k = x - lr * lam_k * x, k + 1
    print(f"kappa = {kappa:5d}  |x| < 1e-6까지 {k:5d} step  축소율 (kappa-1)/(kappa+1) = {(kappa - 1) / (kappa + 1):.4f}")

# %% cell 2 — Armijo backtracking line search (Rosenbrock)
def f(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
def g(p):
    return np.array([-2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] ** 2), 200 * (p[1] - p[0] ** 2)])

def run(steps, lr=None, c=1e-4):
    p, evals = np.array([-1.2, 1.0]), 0
    for _ in range(steps):
        d, t = g(p), 1.0 if lr is None else lr
        while lr is None and f(p - t * d) > f(p) - c * t * d @ d:   # Armijo 조건
            t, evals = t / 2, evals + 1
        p = p - t * d
    return p, f(p), evals

for name, lr in [("고정 1.0e-3", 1e-3), ("고정 2.0e-3", 2e-3), ("고정 2.5e-3", 2.5e-3), ("Armijo", None)]:
    p, fp, ev = run(5000, lr)
    print(f"{name:11s} 5000 step 후 x = {p.round(4)}  f = {fp:.2e}  추가 함수 평가 {ev}")

# %% cell 3 — mini-batch gradient의 잡음: 기댓값과 분산
rng = np.random.default_rng(0)
N = 1000
X = rng.normal(size=(N, 2))                          # (N, D)
y = X @ np.array([2.0, -1.0]) + 0.5 * rng.normal(size=N)
w = np.zeros(2)
full = X.T @ (X @ w - y) / N                          # 전체 gradient (D,)

for B in [1, 10, 100]:
    grads = []
    for _ in range(2000):
        idx = rng.choice(N, size=B, replace=False)
        grads.append(X[idx].T @ (X[idx] @ w - y[idx]) / B)
    grads = np.array(grads)                           # (2000, D)
    bias = np.linalg.norm(grads.mean(axis=0) - full)
    var = grads.var(axis=0).sum()
    print(f"B = {B:3d}  |평균 - 전체 grad| = {bias:.3f}  분산 합 = {var:7.3f}  B x 분산 = {B * var:.2f}")

# %% cell 4 — SGD: 고정 학습률의 잡음 바닥과 감소하는 학습률
w_star = np.linalg.lstsq(X, y, rcond=None)[0]
def excess(w):                                        # L(w) - L(w*)
    return 0.5 * np.mean((X @ w - y) ** 2) - 0.5 * np.mean((X @ w_star - y) ** 2)

for name, sched in [("고정 0.1 ", lambda k: 0.1), ("고정 0.01", lambda k: 0.01),
                    ("0.1/(1+k/100)", lambda k: 0.1 / (1 + k / 100))]:
    rng = np.random.default_rng(1)
    w, hist = np.zeros(2), []
    for k in range(20000):
        i = rng.integers(N)                           # B = 1
        w = w - sched(k) * X[i] * (X[i] @ w - y[i])
        hist.append(excess(w))
    print(f"{name:14s} 초과 손실: step 1000 {hist[999]:.1e}  step 20000 {hist[-1]:.1e}"
          f"  마지막 2000 step 평균 {np.mean(hist[-2000:]):.1e}")

# %% cell 5 — GD · heavy ball · Nesterov: 2차원 궤적 비교
Hq = np.array([1.0, 50.0])                            # Hessian 고윳값: mu = 1, L = 50
def trajectory(lr, beta, nesterov=False, steps=300):
    x, v, path = np.array([10.0, 1.0]), np.zeros(2), []
    for _ in range(steps):
        look = x + beta * v if nesterov else x       # Nesterov: 앞서 본 점의 gradient
        v = beta * v - lr * Hq * look
        x = x + v
        path.append(x.copy())
    return np.array(path)                             # (steps, 2)

r = (np.sqrt(50) - 1) / (np.sqrt(50) + 1)             # (sqrt(kappa) - 1) / (sqrt(kappa) + 1)
for name, lr, beta, nes in [("GD lr=1/L", 1 / 50, 0, False), ("GD lr=1.9/L", 1.9 / 50, 0, False),
                            ("HB 0.9", 1 / 50, 0.9, False), ("Nesterov 0.9", 1 / 50, 0.9, True),
                            ("HB 최적", 4 / (np.sqrt(50) + 1) ** 2, r**2, False),
                            ("Nesterov 최적", 1 / 50, r, True)]:
    P = trajectory(lr, beta, nes)
    f_val = 0.5 * np.sum(Hq * P**2, axis=1)
    hit = int(np.argmax(f_val < 1e-6)) + 1 if (f_val < 1e-6).any() else "-"
    flips = [int(np.sum(np.diff(np.sign(P[:, j])) != 0)) for j in (0, 1)]
    print(f"{name:13s} lr {lr:.4f} beta {beta:.3f}  f<1e-6 도달 {hit!s:>3} step  "
          f"부호 바뀜 x {flips[0]:2d}회, y {flips[1]:3d}회")
