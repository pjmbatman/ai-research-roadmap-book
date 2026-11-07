"""10.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/optimization/adaptive-methods/cells.py
"""
# %% cell 1 — AdaGrad · RMSProp · Adam 구현과 첫 step의 크기
import numpy as np

def make(name, lr, beta=0.9, rho=0.999, eps=1e-8, bias_correction=True):
    s = {"m": 0.0, "v": 0.0, "t": 0}
    def step(x, g):
        s["t"] += 1
        if name == "momentum":
            s["m"] = beta * s["m"] + g
            return x - lr * s["m"]
        if name == "adagrad":
            s["v"] = s["v"] + g * g                          # 제곱 gradient의 누적합
            return x - lr * g / (np.sqrt(s["v"]) + eps)
        s["v"] = rho * s["v"] + (1 - rho) * g * g            # 제곱 gradient의 EMA
        if name == "rmsprop":
            return x - lr * g / (np.sqrt(s["v"]) + eps)
        s["m"] = beta * s["m"] + (1 - beta) * g              # adam: gradient의 EMA
        m, v = s["m"], s["v"]
        if bias_correction:
            m, v = m / (1 - beta ** s["t"]), v / (1 - rho ** s["t"])
        return x - lr * m / (np.sqrt(v) + eps)
    return step

g = np.array([0.01, 100.0])                                  # 두 좌표의 gradient가 10^4배 다르다
for label, opt in [("momentum", make("momentum", 0.1)), ("AdaGrad", make("adagrad", 0.1)),
                   ("RMSProp(0.99)", make("rmsprop", 0.1, rho=0.99)), ("Adam", make("adam", 0.1)),
                   ("Adam 보정 없음", make("adam", 0.1, bias_correction=False))]:
    print(f"{label:14s} 첫 step 이동량 = {np.abs(opt(np.zeros(2), g))}")

# %% cell 2 — 2차원 지형에서 궤적 비교: 축 정렬 vs 회전된 골짜기
def landscape(theta):
    c, s_ = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s_], [s_, c]])
    return R @ np.diag([1.0, 100.0]) @ R.T                   # (2, 2) Hessian, kappa = 100

def steps_to(name, lr, H, tol=1e-6, max_steps=3000):
    opt, x = make(name, lr), np.array([0.8, -0.5])
    for k in range(1, max_steps + 1):
        x = opt(x, H @ x)
        if not np.isfinite(x).all():
            return np.inf
        if 0.5 * x @ H @ x < tol:
            return k
    return np.inf

lrs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
with np.errstate(over="ignore", invalid="ignore"):
    for tname, theta in [("축 정렬  ", 0.0), ("45도 회전", np.pi / 4)]:
        H = landscape(theta)
        for name in ["momentum", "adagrad", "rmsprop", "adam"]:
            ks = [steps_to(name, lr, H) for lr in lrs]       # optimizer마다 학습률을 따로 고른다
            i = int(np.argmin(ks))
            print(f"{tname} {name:8s} 최선 lr {lrs[i]:<6} f < 1e-6까지 {ks[i]:5.0f} step")

# %% cell 3 — L2 정규화와 decoupled weight decay
a = np.array([0.01, 1.0, 100.0])                             # 좌표별 곡률(gradient 스케일)
grad = lambda w: a * (w - 1.0)                               # L(w) = sum a_i (w_i - 1)^2 / 2
lam, lr = 1.0, 0.01

w_l2, w_wd = np.zeros(3), np.zeros(3)
for _ in range(2000):                                        # SGD: 두 방식이 같은가?
    w_l2 = w_l2 - lr * (grad(w_l2) + lam * w_l2)
    w_wd = w_wd - lr * grad(w_wd) - lr * lam * w_wd
print(f"SGD   L2 vs decoupled 최대 차이: {np.abs(w_l2 - w_wd).max():.1e}")

def adam(decoupled, steps=20000, b1=0.9, b2=0.999, eps=1e-8):
    w, m, v = np.zeros(3), np.zeros(3), np.zeros(3)
    for t in range(1, steps + 1):
        g = grad(w) if decoupled else grad(w) + lam * w      # L2: 감쇠 항이 gradient에 들어간다
        m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g * g
        u = (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        w = w - lr * u - (lr * lam * w if decoupled else 0.0)
    return w

print("L2 정규화 문제의 해 a/(a+lam):", (a / (a + lam)).round(4))
print("Adam + L2              :", adam(False).round(4))
print("AdamW (decoupled)      :", adam(True).round(4))

# %% cell 4 — 학습률 스케줄: step · cosine · warm-up + cosine · one-cycle
T, lr_max, warm = 1000, 0.1, 50
def step_decay(k):
    return lr_max * 0.1 ** (k // 400)
def cosine(k):
    return 0.5 * lr_max * (1 + np.cos(np.pi * k / T))
def warmup_cosine(k):
    if k < warm:
        return lr_max * (k + 1) / warm
    return 0.5 * lr_max * (1 + np.cos(np.pi * (k - warm) / (T - warm)))
def one_cycle(k, peak=300):                                  # 선형 증가 후 선형 감소(단순화한 형태)
    return lr_max * (0.1 + 0.9 * k / peak) if k < peak else lr_max * (1 - (k - peak) / (T - peak))

print("step   " + "".join(f"{k:>9d}" for k in [0, 25, 50, 300, 400, 500, 800, 999]))
for name, fn in [("step", step_decay), ("cosine", cosine), ("warm+cos", warmup_cosine), ("1cycle", one_cycle)]:
    print(f"{name:8s}" + "".join(f"{fn(k):9.4f}" for k in [0, 25, 50, 300, 400, 500, 800, 999]))

# %% cell 5 — gradient clipping과 gradient accumulation
rng = np.random.default_rng(0)
X, y = rng.normal(size=(64, 3)), rng.normal(size=64)        # (N, D), (N,)
w = np.ones(3)
grad_of = lambda Xb, yb: Xb.T @ (Xb @ w - yb) / len(yb)     # 평균 손실의 gradient

g = grad_of(X, y)
for max_norm in [10.0, 0.5]:
    scale = min(1.0, max_norm / np.linalg.norm(g))           # global norm clipping
    gc = g * scale
    cos = gc @ g / (np.linalg.norm(gc) * np.linalg.norm(g))
    print(f"clip {max_norm:4.1f}: |g| {np.linalg.norm(g):.3f} -> {np.linalg.norm(gc):.3f}, 방향 cos = {cos:.3f}")

parts = [slice(0, 16), slice(16, 32), slice(32, 48), slice(48, 64)]   # micro-batch 4개
acc_mean = sum(grad_of(X[p], y[p]) for p in parts) / len(parts)
acc_sum = sum(grad_of(X[p], y[p]) for p in parts)
print("누적 후 평균 == 전체 batch gradient :", np.allclose(acc_mean, g))
print("누적 후 합(나누기 생략) / 전체      :", (np.linalg.norm(acc_sum) / np.linalg.norm(g)).round(3))
uneven = [slice(0, 8), slice(8, 64)]                         # 크기가 다른 micro-batch
print("크기가 다른 micro-batch의 단순 평균 오차:", np.linalg.norm(sum(grad_of(X[p], y[p]) for p in uneven) / 2 - g).round(3))
