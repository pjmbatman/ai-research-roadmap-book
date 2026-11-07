"""10.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/optimization/nonconvex-landscape/cells.py
"""
# %% cell 1 — 무작위 대칭 행렬이 양의 정부호일 확률, 안장점 근처의 정체
import numpy as np

rng = np.random.default_rng(0)
for n in [1, 2, 4, 6, 8]:
    A = rng.normal(size=(20000, n, n))
    S = (A + A.transpose(0, 2, 1)) / 2                    # (20000, n, n) 무작위 대칭 행렬
    pd = np.mean(np.linalg.eigvalsh(S)[:, 0] > 0)         # 최소 고윳값 > 0
    print(f"n = {n}  모든 고윳값이 양수일 비율 = {pd:.4f}")

def grad(p):                                              # f = x^2/2 - y^2/2 + y^4/4, 원점은 안장점
    return np.array([p[0], -p[1] + p[1] ** 3])
for eps in [1e-1, 1e-3, 1e-6, 1e-9, 0.0]:
    p, k, g_min = np.array([1.0, eps]), 0, np.inf
    while abs(p[1]) < 0.5 and k < 1000:                    # y가 0.5를 넘으면 안장점을 벗어난 것
        p, k = p - 0.1 * grad(p), k + 1
        g_min = min(g_min, np.linalg.norm(grad(p)))
    print(f"y0 = {eps:7.0e}  안장점을 벗어나기까지 {k:4d} step  (도중 최소 |grad| = {g_min:.1e})")

# %% cell 2 — 은닉 뉴런 8개 MLP: 초기화에 따른 최종 손실
xs = np.linspace(-2, 2, 64)                                 # (N,) 입력
ys = np.sin(3 * xs)                                         # (N,) 목표
H = 8                                                       # 은닉 뉴런 수, theta = (w1, b1, w2, b2): 3H + 1개

def loss_grad(th):
    w1, b1, w2, b2 = th[:H], th[H:2 * H], th[2 * H:3 * H], th[3 * H]
    a = np.tanh(np.outer(xs, w1) + b1)                      # (N, H) 은닉 활성
    r = a @ w2 + b2 - ys                                    # (N,) 잔차
    da = np.outer(r, w2) * (1 - a**2) / len(xs)             # (N, H)
    g = np.concatenate([xs @ da, da.sum(0), a.T @ r / len(xs), [r.mean()]])
    return 0.5 * np.mean(r**2), g

def train(th, lr=0.2, steps=3000):
    for _ in range(steps):
        th = th - lr * loss_grad(th)[1]
    return th

sols = [train(np.random.default_rng(s).normal(size=3 * H + 1)) for s in range(20)]
finals = np.array([loss_grad(t)[0] for t in sols])
print(f"무작위 초기화 시드 20개: 최종 손실 최소 {finals.min():.5f}  중앙값 {np.median(finals):.5f}  최대 {finals.max():.5f}")
th_c = train(np.full(3 * H + 1, 0.5))                       # 모든 파라미터를 같은 값으로 초기화
print(f"모든 파라미터를 0.5로 초기화: 최종 손실 {loss_grad(th_c)[0]:.5f}, 서로 다른 w1 값 {np.unique(th_c[:H].round(8)).size}개")

# %% cell 3 — 같은 함수인데 sharpness가 다르다, SAM 목적함수
xr = np.linspace(-1, 1, 41)
yr = 2.0 * np.maximum(0, xr)                                 # w1 * w2 = 2이면 정확히 맞는다
def L2(w):                                                   # 모델 w2 * relu(w1 * x)
    return 0.5 * np.mean((w[1] * np.maximum(0, w[0] * xr) - yr) ** 2)
def num_hess(fun, w, h=1e-4):
    E = np.eye(len(w)) * h
    return np.array([[(fun(w + E[i] + E[j]) - fun(w + E[i] - E[j]) - fun(w - E[i] + E[j]) + fun(w - E[i] - E[j]))
                      / (4 * h * h) for j in range(len(w))] for i in range(len(w))])
for a in [0.1, 1.0, np.sqrt(2), 10.0]:
    w = np.array([a, 2 / a])                                 # (w1, w2) = (a, 2/a): 모두 같은 함수
    print(f"w1 = {a:5.2f}, w2 = {2 / a:5.2f}  손실 {L2(w):.0e}  Hessian 최대 고윳값 {np.linalg.eigvalsh(num_hess(L2, w))[-1]:6.2f}")

def L1(w):                                                   # w = -1: 깊고 날카로운 골, w = 1.5: 얕고 평평한 골
    return 0.05 * w**2 + 1 - np.exp(-((w + 1) / 0.15) ** 2) - 0.8 * np.exp(-((w - 1.5) / 0.8) ** 2)
grid = np.linspace(-3, 4, 1401)                              # (G,)
for rho in [0.0, 0.1, 0.2]:
    sam = np.array([L1(np.linspace(w - rho, w + rho, 41)).max() for w in grid])   # max_{|e|<=rho} L(w+e)
    print(f"rho = {rho}: 반경 안 최대 손실  w=-1에서 {sam[400]:.3f}, w=1.5에서 {sam[900]:.3f}  ->  최소점 w = {grid[np.argmin(sam)]:.2f}")

# %% cell 4 — 큰 학습률은 sharpness가 2/lr 이하인 최솟점만 남긴다
def grad_uv(p):                                              # f(u, v) = (u v - 1)^2 / 2
    return (p[0] * p[1] - 1) * np.array([p[1], p[0]])
def sharpness(p):                                            # Hessian 최대 고윳값
    u, v = p
    return np.linalg.eigvalsh(np.array([[v * v, 2 * u * v - 1], [2 * u * v - 1, u * u]]))[-1]

for lr in [0.01, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25]:
    p = np.array([0.2, 4.0])                                 # 가장 가까운 최솟점의 sharpness는 약 16
    with np.errstate(all="ignore"):
        for _ in range(5000):
            p = p - lr * grad_uv(p)
    if np.isfinite(p).all():
        print(f"lr = {lr:4}  2/lr = {2 / lr:6.1f}  최종 f = {(p[0] * p[1] - 1) ** 2 / 2:.0e}"
              f"  도착점 ({p[0]:+.2f}, {p[1]:+.2f})  sharpness {sharpness(p):5.2f}")
    else:
        print(f"lr = {lr:4}  2/lr = {2 / lr:6.1f}  발산")

# %% cell 5 — 선형 보간 경로의 손실 장벽과 Hessian 최대 고윳값
def path_losses(a, b, ts=np.linspace(0, 1, 11)):
    return np.array([loss_grad((1 - t) * a + t * b)[0] for t in ts])   # (T,)

perm = np.random.default_rng(1).permutation(H)
idx = np.concatenate([perm, H + perm, 2 * H + perm, [3 * H]])
th_a, th_b = sols[0], sols[1]
th_p = th_a[idx]                                            # 은닉 뉴런 순서만 바꾼 같은 함수
for name, b in [("시드 0 -> 시드 1     ", th_b), ("시드 0 -> 뉴런 순서 바꾼 시드 0", th_p)]:
    Lp = path_losses(th_a, b)
    print(f"{name}  양 끝 {Lp[0]:.4f}, {Lp[-1]:.4f}  경로 최대 {Lp.max():.4f}  장벽 {Lp.max() - max(Lp[0], Lp[-1]):.4f}")

def top_eig(th, iters=100, h=1e-5):
    v = np.random.default_rng(0).normal(size=th.size)
    for _ in range(iters):                                  # power iteration, Hv는 gradient 차분으로
        v = v / np.linalg.norm(v)
        Hv = (loss_grad(th + h * v)[1] - loss_grad(th - h * v)[1]) / (2 * h)
        lam = v @ Hv
        v = Hv
    return lam
for s in [0, 1]:
    print(f"시드 {s} 해의 Hessian 최대 고윳값 {top_eig(sols[s]):.3f}   (학습률 0.2의 안정 한계 2/lr = 10)")
