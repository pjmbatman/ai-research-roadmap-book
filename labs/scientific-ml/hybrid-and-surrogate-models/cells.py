"""24.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

모든 셀은 NumPy만 쓰는 브라우저 셀이다(exec="browser", session="hybrid").
실행: uv run --project labs python labs/scientific-ml/hybrid-and-surrogate-models/cells.py
"""
# %% cell 1 — 문제 정의: Arrhenius 반응 속도, 틀린 문헌값, GP 도구
import numpy as np

R_J = 8.314                                          # 기체 상수, J/(mol K)
def ln_rate(temp_k, ln_a, ea_j):                     # Arrhenius 식: ln k = ln A - Ea / (R T)
    return ln_a - ea_j / (R_J * temp_k)

def gp(x_tr, y_tr, x_te, ell, sf=1.0, sn=0.05):      # GP 회귀(13.7장): 사후 평균·분산, 평균은 데이터 평균
    k = lambda a, b: sf**2 * np.exp(-0.5 * (a[:, None] - b[None, :])**2 / ell**2)
    L = np.linalg.cholesky(k(x_tr, x_tr) + sn**2 * np.eye(x_tr.size))
    v = np.linalg.solve(L, k(x_tr, x_te))            # (N, M)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr - y_tr.mean()))
    return y_tr.mean() + k(x_te, x_tr) @ alpha, sf**2 - np.sum(v**2, axis=0)

truth = lambda t: ln_rate(t, 18.0, 60e3)             # 참 과정(실험자는 모른다)
physics = lambda t: ln_rate(t, 16.0, 55e3)           # 문헌값으로 만든 물리 모델: 식의 구조는 맞고 파라미터가 틀림
rng = np.random.default_rng(0)
temp_tr_k = np.sort(rng.uniform(300, 340, 15))       # (15,) 실험한 온도
ln_k_tr = truth(temp_tr_k) + 0.05 * rng.normal(size=15)
temp_in_k, temp_out_k = np.linspace(305, 335, 50), np.linspace(380, 420, 50)   # 내삽·외삽 시험점
print(f"학습 범위 {temp_tr_k.min():.0f}–{temp_tr_k.max():.0f} K, ln k {ln_k_tr.min():.2f} … {ln_k_tr.max():.2f}")

# %% cell 2 — 네 가지 모델을 같은 조건에서 비교
X = np.column_stack([np.ones(15), -1 / (R_J * temp_tr_k)])   # (15, 2) ln k는 (ln A, Ea)에 선형
ln_a_fit, ea_fit = np.linalg.lstsq(X, ln_k_tr, rcond=None)[0]
resid_tr = ln_k_tr - physics(temp_tr_k)                       # 물리 모델이 설명하지 못한 부분
models = {
    "데이터 전용 GP      ": lambda t: gp(temp_tr_k, ln_k_tr, t, ell=15.0)[0],
    "물리 전용(문헌값)   ": physics,
    "잔차 학습 하이브리드": lambda t: physics(t) + gp(temp_tr_k, resid_tr, t, ell=15.0)[0],
    "파라미터 학습       ": lambda t: ln_rate(t, ln_a_fit, ea_fit),
}
rmse = lambda f, t: np.sqrt(np.mean((f(t) - truth(t))**2))
for name, f in models.items():
    print(f"{name}  내삽 RMSE {rmse(f, temp_in_k):.3f}   외삽 RMSE {rmse(f, temp_out_k):.3f}")
print(f"추정 Ea = {ea_fit / 1e3:.1f} kJ/mol (참값 60, 문헌값 55)")

# %% cell 3 — 다중 충실도: 싼 시뮬레이션 + 비싼 시뮬레이션
f_hi = lambda x: (6 * x - 2)**2 * np.sin(12 * x - 4)          # 고충실도: 비싸다
f_lo = lambda x: 0.5 * f_hi(x) + 10 * (x - 0.5) - 5           # 저충실도: 싸지만 편향
x_hi, x_lo = np.array([0.0, 0.4, 0.6, 1.0]), np.linspace(0, 1, 11)
x_te = np.linspace(0, 1, 101)
mu_hi_only, _ = gp(x_hi, f_hi(x_hi), x_te, ell=0.2, sf=5.0, sn=1e-3)
mu_lo = lambda x: gp(x_lo, f_lo(x_lo), x, ell=0.1, sf=5.0, sn=1e-3)[0]
A = np.column_stack([mu_lo(x_hi), np.ones(4), x_hi])          # (4, 3) f_hi ≈ rho·f_lo + (a + b x) + δ(x)
rho, a, b = np.linalg.lstsq(A, f_hi(x_hi), rcond=None)[0]
trend = lambda x: rho * mu_lo(x) + a + b * x
mu_mf = trend(x_te) + gp(x_hi, f_hi(x_hi) - trend(x_hi), x_te, ell=0.2, sf=1.0, sn=1e-3)[0]
for name, mu in [("고충실도 4점만      ", mu_hi_only), ("저충실도 그대로     ", f_lo(x_te)),
                 ("저 11점 + 고 4점 결합", mu_mf)]:
    print(f"{name}  RMSE {np.sqrt(np.mean((mu - f_hi(x_te))**2)):.3f}")
print(f"추정 rho = {rho:.2f} (구성상 참값 2)")

# %% cell 4 — 능동 학습: 무작위 vs 최대 분산
sim = lambda x: np.sin(6 * x) + 0.5 * np.sin(17 * x)          # 비싼 시뮬레이터라고 가정
x_cand = np.linspace(0, 1, 201)                               # (201,) 후보 입력
def run(pick, n_total=14, n0=4, seed=1):
    r = np.random.default_rng(seed)
    x_tr = r.uniform(0, 1, n0)                                # 초기 실행 4번
    while x_tr.size < n_total:
        x_tr = np.append(x_tr, pick(x_tr, r))
    mu, _ = gp(x_tr, sim(x_tr), x_cand, ell=0.08, sn=1e-3)
    return np.sqrt(np.mean((mu - sim(x_cand))**2))
random_pick = lambda x_tr, r: r.uniform(0, 1)
maxvar_pick = lambda x_tr, r: x_cand[np.argmax(gp(x_tr, sim(x_tr), x_cand, ell=0.08, sn=1e-3)[1])]
for name, pick in [("무작위   ", random_pick), ("최대 분산", maxvar_pick)]:
    errs = [run(pick, seed=s) for s in range(10)]
    print(f"{name}  14번 실행 후 RMSE  평균 {np.mean(errs):.3f}  최악 {np.max(errs):.3f}")
