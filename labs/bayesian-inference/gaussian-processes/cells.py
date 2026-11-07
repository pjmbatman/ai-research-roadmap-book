"""13.7 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/bayesian-inference/gaussian-processes/cells.py
"""
# %% cell 1 — 사전 표본과 길이척도
import numpy as np

def rbf(a, b, ell=1.0, sf=1.0):                 # (n,), (m,) → (n, m)
    return sf**2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)

x = np.linspace(0, 10, 401)                     # (401,) 입력 격자
rng = np.random.default_rng(0)
for ell in (0.3, 1.0, 3.0):
    K = rbf(x, x, ell) + 1e-6 * np.eye(x.size)  # (401, 401), 수치 안정용 jitter
    L = np.linalg.cholesky(K)
    f = L @ rng.normal(size=(x.size, 500))      # (401, 500) 사전 표본 500개
    zc = np.mean(np.sum(np.diff(np.sign(f), axis=0) != 0, axis=0))
    print(f"ell {ell}: 0을 지나는 횟수 평균 {zc:5.2f}  (이론 10/(pi*ell) = {10 / (np.pi * ell):5.2f})"
          f"  f 표준편차 {f.std():.2f}")

# %% cell 2 — GP 회귀: 사후 평균과 분산
def gp_posterior(xtr, ytr, xte, ell, sf, sn):
    K = rbf(xtr, xtr, ell, sf) + sn**2 * np.eye(xtr.size)   # (N, N)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, ytr))   # (N,) K^{-1} y
    Ks = rbf(xtr, xte, ell, sf)                 # (N, M)
    mean = Ks.T @ alpha                         # (M,)
    v = np.linalg.solve(L, Ks)                  # (N, M)
    var = sf**2 - np.sum(v**2, axis=0)          # (M,) 잠재함수 f의 분산
    return mean, var

f_true = lambda x: np.sin(x) + 0.3 * x          # 숨은 응답 곡선
rng = np.random.default_rng(1)
xtr = np.sort(rng.uniform(0, 6, 8))             # (8,) 비싼 실험 8번
ytr = f_true(xtr) + 0.1 * rng.normal(size=8)
xte = np.array([1.0, 3.0, 5.5, 7.0, 9.0])       # 마지막 둘은 외삽
mu, var = gp_posterior(xtr, ytr, xte, ell=1.0, sf=1.0, sn=0.1)
for x0, m, s in zip(xte, mu, np.sqrt(var)):
    print(f"x = {x0}: 예측 {m:6.2f} ± {1.96 * s:4.2f}   참값 {f_true(x0):5.2f}")

# %% cell 3 — 커널과 함수의 매끄러움
def matern12(r, ell=1.0):                       # r = |x - x'|
    return np.exp(-r / ell)
def matern52(r, ell=1.0):
    s = np.sqrt(5) * r / ell
    return (1 + s + s**2 / 3) * np.exp(-s)
def periodic(r, ell=1.0, period=2.0):
    return np.exp(-2 * np.sin(np.pi * r / period) ** 2 / ell**2)
kernels = {"RBF": lambda r: np.exp(-0.5 * r**2), "Matern-1/2": matern12,
           "Matern-5/2": matern52, "periodic(p=2)": periodic}
print("증분 분산 E[(f(x+h) - f(x))^2] = 2(k(0) - k(h))")
print(f"{'kernel':<14} {'h=0.1':>9} {'h=0.01':>9} {'ratio':>6} {'k(r=2)':>7}")
for name, k in kernels.items():
    inc = [2 * (k(0.0) - k(h)) for h in (0.1, 0.01)]     # 증분의 분산
    print(f"{name:<14} {inc[0]:9.2e} {inc[1]:9.2e} {inc[1] / inc[0]:6.3f} {k(2.0):7.3f}")

# %% cell 4 — 로그 주변우도로 길이척도 고르기
def log_ml_terms(xtr, ytr, ell, sf, sn):
    K = rbf(xtr, xtr, ell, sf) + sn**2 * np.eye(xtr.size)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, ytr))
    fit = -0.5 * ytr @ alpha                    # 데이터 적합 항
    complexity = -np.sum(np.log(np.diag(L)))    # -0.5 log det K: 복잡도 벌점
    return fit, complexity, fit + complexity - 0.5 * xtr.size * np.log(2 * np.pi)

rng = np.random.default_rng(2)
xtr = np.sort(rng.uniform(0, 10, 30))           # (30,)
ytr = f_true(xtr) - f_true(xtr).mean() + 0.1 * rng.normal(size=30)
for ell in (0.05, 0.3, 1.0, 3.0, 10.0):
    fit, cx, lml = log_ml_terms(xtr, ytr, ell, sf=1.0, sn=0.1)
    print(f"ell {ell:5}: 적합 {fit:8.1f}  복잡도 {cx:7.1f}  log 주변우도 {lml:7.1f}")

# %% cell 5 — 하이퍼파라미터 최적화
from scipy.optimize import minimize

def neg_lml(log_params):                        # log(ell, sf, sn)를 최적화해 양수 제약을 없앤다
    ell, sf, sn = np.exp(log_params)
    return -log_ml_terms(xtr, ytr, ell, sf, sn)[2]

best = None
for start in ([0.0, 0.0, -1.0], [-2.0, 0.0, -1.0], [2.0, 0.0, -1.0]):   # 여러 시작점
    res = minimize(neg_lml, start, method="L-BFGS-B", bounds=[(-4, 4)] * 3)
    if best is None or res.fun < best.fun:
        best = res
ell, sf, sn = np.exp(best.x)
print(f"최적: ell {ell:.2f}  sf {sf:.2f}  sn {sn:.3f}  (참 잡음 sd 0.1)  log 주변우도 {-best.fun:.1f}")
