"""25.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/scientific-discovery/equation-discovery/cells.py
"""
# %% cell 1 — SINDy: 후보 함수 사전과 희소 회귀
import numpy as np
from scipy.integrate import solve_ivp

a, b, c, d = 1.0, 0.5, 0.2, 0.6                       # 로트카–볼테라: 먹이 x, 포식자 y
lv = lambda t, s: [a * s[0] - b * s[0] * s[1], c * s[0] * s[1] - d * s[1]]
dt = 0.01
t = np.arange(0, 20, dt)                              # (T,)
X = solve_ivp(lv, (0, 20), [4.0, 2.0], t_eval=t, rtol=1e-10, atol=1e-10).y.T   # (T, 2) 상태 궤적

names = ["1", "x", "y", "x^2", "xy", "y^2"]
def library(X):                                       # (T, 2) -> (T, 6) 후보 함수 사전
    x, y = X[:, 0], X[:, 1]
    return np.stack([np.ones_like(x), x, y, x**2, x * y, y**2], axis=1)

def stlsq(Theta, dX, thr=0.05, iters=10):             # 순차적 임계 최소제곱
    Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]    # (6, 2)
    for _ in range(iters):
        small = np.abs(Xi) < thr
        Xi[small] = 0
        for j in range(dX.shape[1]):                  # 남은 항만으로 다시 최소제곱
            keep = ~small[:, j]
            Xi[keep, j] = np.linalg.lstsq(Theta[:, keep], dX[:, j], rcond=None)[0]
    return Xi

dX = np.gradient(X, dt, axis=0)                       # (T, 2) 유한차분 미분
Xi_clean = stlsq(library(X), dX)
for j, v in enumerate(["dx/dt", "dy/dt"]):
    print(v, "=", " ".join(f"{Xi_clean[i, j]:+.3f} {n}" for i, n in enumerate(names) if Xi_clean[i, j] != 0))

# %% cell 2 — 잡음과 미분 추정
from scipy.signal import savgol_filter

Xi_true = np.zeros((6, 2))
Xi_true[[1, 4], 0] = [a, -b]                          # dx/dt = a x - b xy
Xi_true[[2, 4], 1] = [-d, c]                          # dy/dt = -d y + c xy

def score(Xi):                                        # 잘못 들어간 항, 빠진 항, 계수 최대 오차
    extra = np.sum((Xi != 0) & (Xi_true == 0))
    missing = np.sum((Xi == 0) & (Xi_true != 0))
    return f"잘못 들어간 항 {extra:2d}  빠진 항 {missing}  계수 최대 오차 {np.abs(Xi - Xi_true).max():.3f}"

rng = np.random.default_rng(0)
found = {}                                            # 잡음 수준별로 찾은 계수
for level in [0.001, 0.01, 0.05, 0.2]:
    Xn = X + level * X.std(0) * rng.normal(size=X.shape)          # 측정 잡음
    dX_fd = np.gradient(Xn, dt, axis=0)                           # 유한차분: 잡음을 1/dt배로 키운다
    Xs = savgol_filter(Xn, 101, 3, axis=0)                        # 국소 다항식으로 평활
    dX_sg = savgol_filter(Xn, 101, 3, deriv=1, delta=dt, axis=0)  # 같은 다항식의 미분
    found[level] = stlsq(library(Xn), dX_fd), stlsq(library(Xs), dX_sg)
    print(f"잡음 {level:5.3f}  유한차분: {score(found[level][0])}")
    print(f"             평활 미분: {score(found[level][1])}")

# %% cell 3 — 차원 분석: Buckingham π 그룹
from scipy.special import ellipk

dims = {"T": [0, 0, 1], "L": [0, 1, 0], "g": [0, 1, -2],   # [질량, 길이, 시간]의 지수
        "m": [1, 0, 0], "theta0": [0, 0, 0]}               # 주기 T, 줄 길이 L, 중력가속도 g, 질량 m, 진폭 theta0
D = np.array(list(dims.values())).T                         # (3, 5) 차원 행렬
print("변수 5개, 차원 행렬의 rank", np.linalg.matrix_rank(D), "-> π 그룹", 5 - np.linalg.matrix_rank(D), "개")

rep = ["L", "g", "m"]                                       # 반복 변수: 기본 차원 셋을 모두 담는다
D_rep = np.array([dims[v] for v in rep]).T                  # (3, 3)
for q in ["T", "theta0"]:
    e = np.linalg.solve(D_rep, -np.array(dims[q])) + 0.0    # q * L^e0 * g^e1 * m^e2 가 무차원
    print(f"Π = {q} · " + " · ".join(f"{v}^({x:+.1f})" for v, x in zip(rep, e)))

def period_s(L_m, g_mps2, m_kg, theta0):                    # 진자 주기의 정확한 식(타원 적분)
    return 4 * np.sqrt(L_m / g_mps2) * ellipk(np.sin(theta0 / 2) ** 2)

rng = np.random.default_rng(0)
L_m, g_mps2, m_kg = rng.uniform(0.1, 2, 100), rng.uniform(1.6, 25, 100), rng.uniform(0.01, 5, 100)
for theta0 in [0.2, 1.0]:
    T_s = period_s(L_m, g_mps2, m_kg, theta0)               # (100,) 길이·중력·질량이 제각각
    Pi1 = T_s * np.sqrt(g_mps2 / L_m)                       # (100,)
    print(f"theta0 = {theta0}: T 범위 {T_s.min():.2f}–{T_s.max():.2f} s,  Π1 = T·sqrt(g/L) 범위 {Pi1.min():.4f}–{Pi1.max():.4f}")

# %% cell 4 — 작은 기호 회귀: 식을 나열하고 Pareto front 고르기
from itertools import combinations
rng = np.random.default_rng(1)
th_tr = np.sort(rng.uniform(0.05, 1.5, 40))                 # (40,) 학습 구간 진폭 [rad]
y_tr = 4 * ellipk(np.sin(th_tr / 2) ** 2) * (1 + 0.001 * rng.normal(size=40))   # Π1, 0.1% 측정 잡음
th_te = np.linspace(1.6, 2.4, 9)                            # 외삽 구간
y_te = 4 * ellipk(np.sin(th_te / 2) ** 2)

basis = {"1": (np.ones_like, 1), "θ": (lambda t: t, 1), "θ^2": (lambda t: t**2, 3),
         "θ^4": (lambda t: t**4, 3), "sin θ": (np.sin, 2), "cos θ": (np.cos, 2),
         "exp θ": (np.exp, 2)}                              # 이름: (함수, 식 트리의 노드 수)
def design(terms, th):                                      # (len(th), len(terms)) 설계 행렬
    return np.stack([f(th) for f, _ in (basis[n] for n in terms)], 1)

def pareto(names):                                          # 항 1–3개짜리 선형 결합을 모두 나열
    rows = []
    for k in (1, 2, 3):
        for terms in combinations(names, k):
            A = design(terms, th_tr)
            w = np.linalg.lstsq(A, y_tr, rcond=None)[0]
            ext = design(terms, th_te) @ w - y_te
            cx = sum(basis[n][1] + 1 for n in terms)        # 복잡도: 노드 수 + 계수
            rows.append((cx, np.sqrt(np.mean((A @ w - y_tr) ** 2)), np.sqrt(np.mean(ext**2)),
                         " + ".join(f"{v:.3f}·{n}" for v, n in zip(w, terms))))
    best = np.inf
    for cx, rmse, ext, f in sorted(rows):                   # 복잡도 순으로, 오차가 5% 넘게 줄 때만 남긴다
        if rmse < 0.95 * best:
            best = rmse
            print(f"  복잡도 {cx:2d}  학습 RMSE {rmse:.4f}  외삽 RMSE {ext:.3f}  {f}")

print("모든 후보 함수"); pareto(list(basis))
print("짝함수만 (θ → -θ 대칭)"); pareto(["1", "θ^2", "θ^4", "cos θ"])

# %% cell 5 — 검증: 새 조건, 보존량, 물리적 극한
def H(X):                                                   # 로트카–볼테라의 보존량
    return c * X[:, 0] - d * np.log(X[:, 0]) + b * X[:, 1] - a * np.log(X[:, 1])

s_new = [1.0, 1.0]                                          # 학습 궤적에 없던 초기 조건
X_new = solve_ivp(lv, (0, 20), s_new, t_eval=t, rtol=1e-10, atol=1e-10).y.T
print("참 모델의 보존량 변동폭 < 1e-6:", np.ptp(H(X_new)) < 1e-6)
for name, Xi in [("잡음 0.05, 평활 미분", found[0.05][1]), ("잡음 0.2,  평활 미분", found[0.2][1])]:
    rhs = lambda _, s: library(np.array([s]))[0] @ Xi       # 찾은 식을 그대로 적분
    Xp = solve_ivp(rhs, (0, 20), s_new, t_eval=t, rtol=1e-8).y.T
    ok = len(Xp) == len(t) and np.all(Xp > 0)
    if ok:
        print(f"{name}: 새 초기 조건 RMSE {np.sqrt(np.mean((Xp - X_new) ** 2)):.3f}  보존량 변동폭 {np.ptp(H(Xp)):.3f}")
    else:
        print(f"{name}: 적분 실패 — 개체 수가 음수가 되거나 발산한다")

def fit_predict(terms, th):                                 # 셀 4와 같은 최소제곱으로 맞춘 뒤 th에서 예측
    w = np.linalg.lstsq(design(terms, th_tr), y_tr, rcond=None)[0]
    return design(terms, np.atleast_1d(th)) @ w

for terms in [("1", "θ^2"), ("1", "θ", "exp θ"), ("1", "θ^4", "cos θ")]:
    f0 = fit_predict(terms, 0.0)[0]
    asym = abs(fit_predict(terms, 1.0)[0] - fit_predict(terms, -1.0)[0])
    print(f"{' + '.join(terms):16s} θ→0 극한 {f0:.3f} (정답 2π = {2 * np.pi:.3f})   |f(1) - f(-1)| = {asym:.3f}")
