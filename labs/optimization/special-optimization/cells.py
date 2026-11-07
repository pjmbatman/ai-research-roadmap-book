"""10.7 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/optimization/special-optimization/cells.py
"""
# %% cell 1 — Pareto front, 가중합 scalarization과 epsilon-제약
import numpy as np

x = np.linspace(0, 1, 101)                                  # (M,) 설계 변수 후보
f1, f2 = x, 1 - x**2                                        # 두 목적(모두 최소화). front가 볼록이 아니다
F = np.stack([f1, f2], axis=1)                              # (M, 2)

def nondominated(F):
    keep = []
    for i, p in enumerate(F):                               # p보다 모든 목적에서 같거나 좋고 하나는 더 좋은 점이 있으면 지배당함
        dominated = np.any(np.all(F <= p, axis=1) & np.any(F < p, axis=1))
        keep.append(not dominated)
    return np.array(keep)

print("Pareto 최적인 후보:", int(nondominated(F).sum()), "/", len(x))
ws = np.linspace(0, 1, 11)
picked = {float(x[np.argmin(w * f1 + (1 - w) * f2)]) for w in ws}   # 가중합 최소화
print("가중합 11개 가중치가 찾은 x:", sorted(picked))
eps_picked = []
for eps in [0.2, 0.4, 0.6, 0.8]:                           # epsilon-제약: f1 <= eps 아래에서 f2 최소화
    ok = f1 <= eps + 1e-12
    eps_picked.append(float(x[ok][np.argmin(f2[ok])]))
print("epsilon-제약이 찾은 x   :", eps_picked)

# %% cell 2 — 이중수준 최적화: ridge 정규화 강도의 hypergradient
rng = np.random.default_rng(0)
D = 30
w_true = rng.normal(size=D) * (rng.random(D) < 0.3)            # (D,) 일부 성분만 0이 아니다
X_tr, X_va = rng.normal(size=(40, D)), rng.normal(size=(200, D))  # (N_tr, D), (N_va, D)
y_tr = X_tr @ w_true + 1.0 * rng.normal(size=40)
y_va = X_va @ w_true + 1.0 * rng.normal(size=200)

def inner(lam):                                                 # 안쪽 문제: ridge 회귀의 해 w*(lam)
    return np.linalg.solve(X_tr.T @ X_tr + lam * np.eye(D), X_tr.T @ y_tr)
def outer(lam):                                                 # 바깥 문제: 검증 손실
    return 0.5 * np.mean((X_va @ inner(lam) - y_va) ** 2)
def hypergrad(lam):                                             # 암시적 미분: dw/dlam = -(X^T X + lam I)^{-1} w
    w = inner(lam)
    dw = -np.linalg.solve(X_tr.T @ X_tr + lam * np.eye(D), w)
    return (X_va.T @ (X_va @ w - y_va) / len(y_va)) @ dw

lam, h = 1.0, 1e-5
print(f"hypergradient: 암시적 미분 {hypergrad(lam):.6f}  유한 차분 {(outer(lam + h) - outer(lam - h)) / (2 * h):.6f}")
s = np.log(0.1)                                                 # s = log lam 위에서 경사하강
for k in range(1, 201):
    s -= 5.0 * np.exp(s) * hypergrad(np.exp(s))                 # dL/ds = lam dL/dlam
    if k in (1, 10, 50, 200):
        print(f"step {k:3d}  lam = {np.exp(s):8.4f}  검증 손실 {outer(np.exp(s)):.4f}")
grid = np.logspace(-3, 3, 61)
best = grid[np.argmin([outer(g) for g in grid])]
print(f"격자 탐색(61점) 최선 lam = {best:.4f}  검증 손실 {outer(best):.4f}   lam = 0.1의 검증 손실 {outer(0.1):.4f}")

# %% cell 3 — 블랙박스 최적화: 격자 vs random search, (1+1)-ES
def f_low(p):                                                    # 사실상 x1만 중요한 2차원 함수, 최소는 x1 = 0.37
    return -np.exp(-((p[..., 0] - 0.37) ** 2) / 0.01) + 0.01 * p[..., 1]

g = np.linspace(0, 1, 3)
grid_pts = np.array([[a, b] for a in g for b in g])              # (9, 2) 3 x 3 격자
rng = np.random.default_rng(0)
rand_best = [f_low(rng.random((9, 2))).min() for _ in range(1000)]  # 9점 random search 1000번
print(f"평가 9번: 격자 최선 {f_low(grid_pts).min():.3f} (시험한 x1 값 {len(g)}개)"
      f"   random 중앙값 {np.median(rand_best):.3f} (시험한 x1 값 9개)")

def rosen(z):                                                    # 10차원 Rosenbrock, 최솟값 0
    return np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (1 - z[:-1]) ** 2)
def one_plus_one_es(f, x, sigma, budget, rng):
    fx, succ = f(x), []
    for _ in range(budget - 1):
        y = x + sigma * rng.normal(size=x.size)                  # 자식 하나
        fy = f(y)
        succ.append(fy <= fx)
        if fy <= fx:
            x, fx = y, fy
        if len(succ) == 20:                                      # 1/5 성공 규칙으로 step 크기 조절
            sigma *= 1.5 if np.mean(succ) > 0.2 else 1 / 1.5
            succ = []
    return fx
for budget in [100, 1000, 10000]:
    es = [one_plus_one_es(rosen, np.zeros(10), 0.5, budget, np.random.default_rng(s)) for s in range(5)]
    rs = [min(rosen(p) for p in np.random.default_rng(s).uniform(-2, 2, (budget, 10))) for s in range(5)]
    print(f"평가 {budget:5d}번  (1+1)-ES 중앙값 {np.median(es):9.3f}   random search(-2..2) 중앙값 {np.median(rs):9.1f}")

# %% cell 4 — 베이지안 최적화: GP 대리모델과 expected improvement
from scipy.stats import norm

def sim(x):                                                      # "비싼 시뮬레이터" 대역: [0, 2]에서 최소화
    return np.sin(3 * x) + 0.5 * np.sin(10 * x) + 0.3 * x
cand = np.linspace(0, 2, 401)                                    # (C,) 후보 점
def kern(a, b, ell=0.15):                                        # RBF 커널, 사전 분산 1
    return np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)
def gp_posterior(X, y, Xs, noise=1e-6):
    m, s = y.mean(), y.std() + 1e-12                             # y를 표준화
    L = np.linalg.cholesky(kern(X, X) + noise * np.eye(len(X)))  # (n, n)
    v = np.linalg.solve(L, kern(X, Xs))                          # (n, C)
    mu = v.T @ np.linalg.solve(L, (y - m) / s)                   # (C,)
    return m + s * mu, s * np.sqrt(np.clip(1 - np.sum(v**2, 0), 1e-12, None))   # 평균, 표준편차
def expected_improvement(mu, sd, best, xi=0.01):                # 최소화 문제의 EI
    z = (best - mu - xi) / sd
    return (best - mu - xi) * norm.cdf(z) + sd * norm.pdf(z)
def run(method, seed, budget=15):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 2, 3)                                     # 처음 3점은 무작위
    while len(X) < budget:
        if method == "BO":
            mu, sd = gp_posterior(X, sim(X), cand)
            x_next = cand[np.argmax(expected_improvement(mu, sd, sim(X).min()))]
        else:
            x_next = rng.uniform(0, 2)
        X = np.append(X, x_next)
    return np.minimum.accumulate(sim(X)) - sim(cand).min()       # 평가 횟수별 regret (budget,)

for method in ["BO", "random"]:
    R = np.array([run(method, s) for s in range(20)])            # (시드, budget)
    print(f"{method:6s} regret 중앙값  평가 5번 {np.median(R[:, 4]):.3f}  10번 {np.median(R[:, 9]):.3f}"
          f"  15번 {np.median(R[:, 14]):.4f}   15번 안에 regret < 0.01: {np.mean(R[:, 14] < 0.01):.2f}")

# %% cell 5 — 이산 문제: knapsack의 LP relaxation과 정수 계획
from itertools import product
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

# 예산 안에서 실험(또는 측정 장비) 고르기: 0-1 knapsack
value = np.array([10, 13, 7, 8, 9, 4, 6, 11.0])                  # (n,) 각 실험의 기대 정보량
cost = np.array([5, 7, 4, 5, 6, 3, 4, 6.0])                       # (n,) 비용 [백만 원]
budget = 21.0

lp = linprog(-value, A_ub=cost[None, :], b_ub=[budget], bounds=[(0, 1)] * len(value))   # relaxation: 0 <= x <= 1
print("LP relaxation 해 x =", np.round(lp.x, 3), f" 목적값 {-lp.fun:.3f}")
x_down = np.floor(lp.x + 1e-9)                                    # 내림: 항상 feasible
x_round = np.round(lp.x)                                          # 반올림: feasible이 아닐 수 있다
for name, xr in [("내림", x_down), ("반올림", x_round)]:
    print(f"{name:3s}  x = {xr.astype(int)}  가치 {value @ xr:5.1f}  비용 {cost @ xr:4.1f}  feasible = {cost @ xr <= budget}")

res = milp(-value, constraints=LinearConstraint(cost[None, :], 0, budget),
           integrality=np.ones(len(value)), bounds=Bounds(0, 1))   # 정수 계획(branch and bound)
brute = max(value @ np.array(c) for c in product([0, 1], repeat=len(value)) if cost @ np.array(c) <= budget)
print(f"정수 최적  x = {np.round(res.x).astype(int)}  가치 {-res.fun:5.1f}  (전수 조사 {brute:.1f})"
      f"   relaxation 간극 {-lp.fun - (-res.fun):.3f}")
