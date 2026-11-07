"""14.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/information-and-learning-theory/modern-generalization/cells.py
"""
# %% cell 1 — 최소 노름 회귀의 double descent
import numpy as np

n, D, noise = 40, 200, 0.1                         # 학습 표본 40개, 후보 특징 200개
beta = np.arange(1, D + 1) ** -0.5                 # 참 계수: 앞쪽 특징일수록 중요, (D,)
beta /= np.linalg.norm(beta)

def risk(w, p):                                    # 등방 가우시안 입력에서 시험 MSE의 정확한 값
    return np.sum((w - beta[:p]) ** 2) + np.sum(beta[p:] ** 2) + noise**2

print("  p    최소 노름: 위험   ‖w‖     ridge(λ=1): 위험")
for p in (5, 10, 20, 30, 36, 40, 44, 50, 80, 120, 200):
    out = []
    for seed in range(20):
        r = np.random.default_rng(seed)
        X = r.normal(size=(n, D))                  # (N, D)
        y = X @ beta + noise * r.normal(size=n)
        Xp = X[:, :p]                              # 앞의 p개 특징만 쓰는 모델, (N, p)
        w = np.linalg.pinv(Xp) @ y                 # 최소 노름 최소제곱 해, (p,)
        w_r = Xp.T @ np.linalg.solve(Xp @ Xp.T + 1.0 * np.eye(n), y)   # ridge 해, (p,)
        out.append((risk(w, p), np.linalg.norm(w), risk(w_r, p)))
    m = np.median(out, axis=0)
    print(f"{p:4d}       {m[0]:7.2f}      {m[1]:5.2f}       {m[2]:5.2f}")

# %% cell 2 — 경사하강의 암묵적 편향: 초기값이 어떤 보간 해를 고르는가
rng = np.random.default_rng(0)
X = rng.normal(size=(20, 100))                     # 표본 20개, 파라미터 100개, (N, p)
w_true = rng.normal(size=100) / 10
y = X @ w_true                                     # 잡음 없는 관측, (N,)
w_pinv = np.linalg.pinv(X) @ y                     # 최소 노름 보간 해
lr = 1.0 / np.linalg.norm(X, 2) ** 2               # 안정 조건 η < 2/σ_max²
for name, w in (("0에서 시작", np.zeros(100)), ("무작위 초기값", rng.normal(size=100))):
    for _ in range(3000):
        w = w - lr * X.T @ (X @ w - y)             # 0.5‖Xw - y‖²의 경사하강
    print(f"{name:8s} 학습 잔차 < 1e-8: {np.linalg.norm(X @ w - y) < 1e-8}   ‖w‖ = {np.linalg.norm(w):5.2f}   "
          f"‖w - w_pinv‖ = {np.linalg.norm(w - w_pinv):5.2f}   ‖w - w_true‖ = {np.linalg.norm(w - w_true):5.2f}")
print(f"최소 노름 해     ‖w_pinv‖ = {np.linalg.norm(w_pinv):5.2f}")

# %% cell 3 — 작은 범위의 스케일링 법칙을 멀리 외삽하기
def knn_test_mse(n_train, seed, k=5, noise=0.2):
    r = np.random.default_rng(seed)
    f = lambda x: np.sin(3 * x[:, 0]) * np.cos(3 * x[:, 1])
    x = r.uniform(0, 1, (n_train, 2)); y = f(x) + noise * r.normal(size=n_train)
    xt = r.uniform(0, 1, (300, 2)); yt = f(xt) + noise * r.normal(size=300)
    d2 = ((xt[:, None, :] - x[None, :, :]) ** 2).sum(-1)       # (300, N)
    idx = np.argpartition(d2, k, axis=1)[:, :k]                # 가까운 이웃 k개
    return np.mean((y[idx].mean(1) - yt) ** 2)

Ns = np.array([25, 50, 100, 200, 400, 800, 1600, 3200])
L = np.array([np.mean([knn_test_mse(N, s) for s in range(20)]) for N in Ns])
small = Ns <= 200                                  # 작은 범위에서만 법칙을 맞춘다
b, log_a = np.polyfit(np.log(Ns[small]), np.log(L[small]), 1)  # L = a N^b (바닥 없음)
print("    N   측정 MSE   멱법칙 외삽")
for N, l in zip(Ns, L):
    print(f"{N:5d}   {l:.4f}     {np.exp(log_a) * N**b:.4f}{'' if N <= 200 else '  ← 외삽'}")
print(f"맞춘 지수 b = {b:.3f},  잡음 분산 = {0.2**2:.4f}")

# %% cell 4 — 고차원에서 '새 입력'은 거의 늘 학습 데이터의 convex hull 밖이다
from scipy.optimize import linprog

def in_hull(points, q):                            # q = Σ λ_i x_i, Σ λ_i = 1, λ ≥ 0 이 가능한가
    A = np.vstack([points.T, np.ones(len(points))])
    res = linprog(np.zeros(len(points)), A_eq=A, b_eq=np.append(q, 1.0), bounds=(0, None), method="highs")
    return res.status == 0

rng = np.random.default_rng(0)
print("차원 d   학습 500개의 convex hull 안에 든 새 입력의 비율")
for d in (1, 2, 4, 8, 16):
    train = rng.normal(size=(500, d))              # (N, d)
    test = rng.normal(size=(100, d))               # 같은 분포에서 뽑은 새 입력, (100, d)
    frac = np.mean([in_hull(train, q) for q in test])
    print(f"{d:5d}      {frac:.2f}")
