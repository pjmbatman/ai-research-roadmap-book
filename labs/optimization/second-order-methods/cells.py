"""10.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/optimization/second-order-methods/cells.py
"""
# %% cell 1 — Rosenbrock 함수에서 damped Newton: 감쇠 단계와 2차 수렴 단계
import numpy as np

def f(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
def grad(p):
    return np.array([-2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] ** 2), 200 * (p[1] - p[0] ** 2)])
def hess(p):
    return np.array([[2 - 400 * (p[1] - 3 * p[0] ** 2), -400 * p[0]],
                     [-400 * p[0], 200.0]])

p, x_star = np.array([-1.2, 1.0]), np.ones(2)
for k in range(1, 40):
    g = grad(p)
    d = -np.linalg.solve(hess(p), g)                 # Newton 방향: H d = -g
    t = 1.0
    while f(p + t * d) > f(p) + 1e-4 * t * g @ d:     # Armijo backtracking
        t /= 2
    p = p + t * d
    err = np.linalg.norm(p - x_star)
    if k <= 2 or k >= 16:
        print(f"k = {k:2d}  t = {t:<6}  |x - x*| = {err:.2e}")
    if err < 1e-12:
        break

# %% cell 2 — 이차함수에서 한 step, 비볼록 함수에서 안장점으로 끌려가는 Newton
for kappa in [10.0, 1e3, 1e6]:
    H, b = np.diag([1.0, kappa]), np.array([1.0, 1.0])  # (2, 2), (2,)
    z = np.array([5.0, -3.0])
    z = z - np.linalg.solve(H, H @ z - b)               # Newton 한 step
    print(f"kappa = {kappa:7.0e}  Newton 1 step 후 |grad| = {np.linalg.norm(H @ z - b):.1e}")

def g1(z):
    return z**3 - z + 0.2                               # 10.1장의 기울어진 이중 우물 (y = 0 단면)
def h1(z):
    return 3 * z**2 - 1
for z0 in [0.3, 0.5, 1.5]:
    zn, zg = z0, z0
    for _ in range(200):
        zn = zn - g1(zn) / h1(zn)                       # Newton
        zg = zg - 0.1 * g1(zg)                          # GD
    print(f"z0 = {z0}:  Newton -> {zn:.3f} (f'' = {h1(zn):+.2f})   GD -> {zg:.3f} (f'' = {h1(zg):+.2f})")

# %% cell 3 — 지수 감쇠 모델 적합: Gauss–Newton과 Levenberg–Marquardt
rng = np.random.default_rng(0)
t_s = np.linspace(0, 5, 40)                                  # (N,) 측정 시각 [s]
def model(th):                                               # th = (a, k [1/s], c)
    return th[0] * np.exp(-th[1] * t_s) + th[2]
y = model([2.0, 0.8, 0.5]) + 0.02 * rng.normal(size=t_s.size)
def resid(th):
    return model(th) - y                                     # (N,)
def jac(th):
    e = np.exp(-th[1] * t_s)
    return np.stack([e, -th[0] * t_s * e, np.ones_like(t_s)], axis=1)   # (N, 3)
def cost(th):
    return 0.5 * resid(th) @ resid(th)

def fit(th, lam=None, iters=100):
    th = np.array(th, float)
    for k in range(1, iters + 1):
        J, r = jac(th), resid(th)
        if lam is None:                                      # Gauss-Newton: J step = -r 의 최소제곱해
            th = th + np.linalg.lstsq(J, -r, rcond=None)[0]
        else:                                                # LM: (J^T J + lam I) step = -J^T r
            step = -np.linalg.solve(J.T @ J + lam * np.eye(3), J.T @ r)
            if cost(th + step) < cost(th):
                th, lam = th + step, lam / 10                # 성공: 감쇠를 줄인다
            else:
                lam = lam * 10                               # 실패: 제자리에서 감쇠를 키운다
        if not np.isfinite(cost(th)) or np.linalg.norm(jac(th).T @ resid(th)) < 1e-8:
            break
    return th, k

with np.errstate(all="ignore"):
    for start in [[1.0, 1.0, 0.0], [1.0, 5.0, 0.0], [1.0, 0.05, 0.0]]:
        for name, lam in [("Gauss-Newton", None), ("LM", 1e-2)]:
            th, k = fit(start, lam)
            print(f"시작 k = {start[1]:4}  {name:12s} {k:3d}회  theta = {np.round(th, 3)}  cost = {cost(th):.2e}")

# %% cell 4 — scipy.optimize로 비교하는 CG·BFGS·L-BFGS·Newton(trust region)
from scipy.optimize import minimize, rosen, rosen_der, rosen_hess

for n in [2, 100]:
    x0 = np.tile([-1.2, 1.0], n // 2)                        # (n,) 확장 Rosenbrock의 표준 시작점
    for method in ["CG", "BFGS", "L-BFGS-B", "trust-exact"]:
        res = minimize(rosen, x0, jac=rosen_der, method=method,
                       hess=rosen_hess if method == "trust-exact" else None)
        print(f"n = {n:3d}  {method:11s} 반복 {res.nit:5d}  grad 평가 {res.njev:5d}"
              f"  f = {res.fun:.1e}  x_1 = {res.x[0]:+.3f}")
x0 = np.tile([-1.2, 1.0], 50)
for m in [1, 3, 10, 30]:
    res = minimize(rosen, x0, jac=rosen_der, method="L-BFGS-B", options={"maxcor": m})
    print(f"L-BFGS 기억 m = {m:2d}  반복 {res.nit:4d}  저장 실수 2mn = {2 * m * 100:5d}  (BFGS n^2 = {100**2})")

# %% cell 5 — 정규분포 적합: GD와 자연 gradient, 두 매개화
x = np.random.default_rng(0).normal(3.0, 2.0, size=1000)       # (N,) 관측값
m1, m2 = x.mean(), np.mean(x**2)                                # 충분통계량
def nll(mu, sig):                                               # 표본당 음의 로그우도(상수 제외)
    return np.log(sig) + (m2 - 2 * mu * m1 + mu**2) / (2 * sig**2)
nll_star = nll(m1, x.std())

def run(param, natural, lr, sig, mu=0.0, max_steps=5000):
    for k in range(1, max_steps + 1):
        S = m2 - 2 * mu * m1 + mu**2                            # mean((x - mu)^2)
        g_mu, g_sig = -(m1 - mu) / sig**2, 1 / sig - S / sig**3
        if param == "sigma":                                    # Fisher: diag(1/sig^2, 2/sig^2)
            d_mu, d_sig = (sig**2 * g_mu, sig**2 / 2 * g_sig) if natural else (g_mu, g_sig)
            mu, sig = mu - lr * d_mu, sig - lr * d_sig
        else:                                                   # s = log sig, Fisher: diag(1/sig^2, 2)
            g_s = sig * g_sig                                   # 연쇄법칙
            d_mu, d_s = (sig**2 * g_mu, g_s / 2) if natural else (g_mu, g_s)
            mu, sig = mu - lr * d_mu, sig * np.exp(-lr * d_s)
        if not (np.isfinite(sig) and sig > 0):
            return np.inf
        if nll(mu, sig) - nll_star < 1e-8:
            return k
    return np.inf

lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
with np.errstate(all="ignore"):
    for sig0 in [0.1, 10.0]:
        for param, natural in [("sigma", False), ("log sigma", False), ("sigma", True), ("log sigma", True)]:
            ks = [run(param, natural, lr, sig0) for lr in lrs]
            i = int(np.argmin(ks))
            name = ("자연 gradient " if natural else "GD ") + f"({param})"
            best = f"최선 lr {lrs[i]:<6} {ks[i]:5.0f} step" if np.isfinite(ks[i]) else "격자의 모든 lr이 5000 step 안에 실패"
            print(f"sigma0 = {sig0:4}  {name:24s} {best}")
