"""26.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/physics-and-engineering/design-optimization/cells.py
"""
# %% cell 1 — 다목적 설계: 외팔보의 질량과 처짐
import numpy as np

F_n, Lb_m, E_pa, rho_kgm3, sig_allow_pa = 1000.0, 1.0, 200e9, 7850.0, 160e6   # 강철 외팔보, 끝단 하중
b_m, h_m = [a.ravel() for a in np.meshgrid(np.linspace(0.01, 0.05, 41), np.linspace(0.02, 0.10, 81))]
mass_kg = rho_kgm3 * b_m * h_m * Lb_m                        # (3321,)
defl_mm = 4 * F_n * Lb_m**3 / (E_pa * b_m * h_m**3) * 1e3     # 끝단 처짐 δ = F L³ / (3 E I), I = b h³ / 12
ok = 6 * F_n * Lb_m / (b_m * h_m**2) <= sig_allow_pa          # 뿌리의 굽힘 응력 제약
Fobj = np.stack([mass_kg, defl_mm], 1)[ok]                    # (M, 2) 두 목적, 모두 최소화
front = np.array([not np.any(np.all(Fobj <= p, 1) & np.any(Fobj < p, 1)) for p in Fobj])
bf, hf = b_m[ok][front], h_m[ok][front]
print(f"응력 제약을 만족하는 설계 {ok.sum()}개 중 Pareto 최적 {front.sum()}개")
print(f"그중 폭이 최소(1 cm)이거나 높이가 상한 근처(9.5 cm 이상)인 설계: {np.sum(np.isclose(bf, 0.01) | (hf >= 0.095))}개")
for d_max in [8.0, 2.0, 1.0, 0.3]:                            # epsilon-제약: 처짐 <= d_max에서 질량 최소
    feas = Fobj[:, 1] <= d_max
    if not feas.any():
        print(f"처짐 ≤ {d_max:3.1f} mm: 설계 범위 안에 해 없음")
        continue
    i = np.flatnonzero(feas)[np.argmin(Fobj[feas, 0])]
    print(f"처짐 ≤ {d_max:3.1f} mm: 폭 {b_m[ok][i] * 100:.1f} cm, 높이 {h_m[ok][i] * 100:.1f} cm, 질량 {Fobj[i, 0]:5.2f} kg")

# %% cell 2 — adjoint 기울기와 유한차분 기울기
n, L_m, q_wm = 50, 1.0, 1.0                       # 요소 수, 막대 길이, 단위 길이당 발열 [W/m]
h = L_m / n                                       # 요소 길이 [m]

def stiffness(k):                                 # 1차 유한요소 전도 행렬, x=0의 T=0인 절점은 뺀다
    main = np.append(k[:-1] + k[1:], k[-1]) / h   # (n,)
    return np.diag(main) - np.diag(k[1:] / h, 1) - np.diag(k[1:] / h, -1)   # (n, n)

f = np.full(n, q_wm * h); f[-1] /= 2              # (n,) 절점 열부하, 오른쪽 끝은 단열
def peak_temp(k):                                  # 목적: 단열된 끝의 온도 J = T(L)
    return np.linalg.solve(stiffness(k), f)[-1]

def grad_adjoint(k):                               # 해석 1회 + adjoint 해석 1회
    K = stiffness(k)
    T = np.concatenate([[0.0], np.linalg.solve(K, f)])               # (n+1,)
    lam = np.concatenate([[0.0], np.linalg.solve(K.T, np.eye(n)[-1])])   # K^T λ = ∂J/∂T
    return -np.diff(lam) * np.diff(T) / h          # (n,) dJ/dk_e = −λ^T (∂K/∂k_e) T

k0 = np.ones(n)                                    # 균일 열전도도 [W/(m·K)]
g_adj = grad_adjoint(k0)
g_fd = np.array([(peak_temp(k0 + 1e-6 * np.eye(n)[e]) - peak_temp(k0)) / 1e-6 for e in range(n)])
print(f"J = T(L) = {peak_temp(k0):.4f} K (해석해 q L²/(2k) = 0.5)")
print("adjoint와 유한차분의 최대 상대 차이 < 1e-5:", bool(np.max(np.abs(g_adj - g_fd)) / np.max(np.abs(g_adj)) < 1e-5))
print(f"선형계 풀이 횟수: adjoint 2회, 전진 유한차분 {n + 1}회")

# %% cell 3 — 기울기 기반 설계: 재료 예산 아래에서 최고 온도 줄이기
V = 1.0                                            # 재료 예산 Σ k_e h = V
k = np.full(n, V / L_m)
for it in range(30):
    B = -grad_adjoint(k) / h                       # (n,) 재료 단위당 온도 감소 효과 (>= 0)
    lo, hi = 1e-9, 1e9
    while hi / lo > 1 + 1e-10:                     # 예산을 맞추는 Lagrange 승수를 이분법으로 찾는다
        mid = np.sqrt(lo * hi)
        k_new = np.clip(k * np.sqrt(B / mid), np.maximum(0.01, 0.5 * k), 2 * k)
        lo, hi = (mid, hi) if k_new.sum() * h > V else (lo, mid)
    k = k_new
    if it in (0, 1, 4, 29):
        print(f"반복 {it + 1:2d}  J = {peak_temp(k):.5f} K")
x = (np.arange(n) + 0.5) * h
k_star = np.sqrt(L_m - x); k_star *= V / (k_star.sum() * h)   # 연속 문제의 최적해 k ∝ sqrt(L − x)
print(f"연속 문제의 최적값 4qL³/(9V) = {4 * q_wm * L_m**3 / (9 * V):.5f} K")
print("찾은 설계가 k ∝ sqrt(L − x)와 일치(최대 차이 < 1e-6):", bool(np.abs(k - k_star).max() < 1e-6))

# %% cell 4 — 대리모델 최적화의 가짜 최적해와 재검증
design = lambda p: (lambda kk: kk * V / (kk.sum() * h))((L_m - x + 0.02) ** p)   # 형상 파라미터 p 하나
P = [0.0, 0.1, 0.2, 0.3]                                       # 고충실도 해석을 한 설계
Js = [peak_temp(design(p)) for p in P]
grid = np.linspace(0.0, 1.5, 151)
coef = np.polyfit(P, Js, 3)                                    # 3차 다항식 대리모델
p_s = grid[np.argmin(np.polyval(coef, grid))]
print(f"대리모델 최적 p = {p_s:.2f}: 예측 J = {np.polyval(coef, p_s):.5f} K, 재검증 J = {peak_temp(design(p_s)):.5f} K")
for it in range(3):                                            # 재검증 결과를 데이터에 넣고 탐색 범위를 제한한다
    lo, hi = min(P), max(P)
    cand = np.linspace(max(0.0, lo - 0.5 * (hi - lo)), hi + 0.5 * (hi - lo), 301)
    coef = np.polyfit(P, Js, 3)
    p_new = cand[np.argmin(np.polyval(coef, cand))]
    P.append(p_new); Js.append(peak_temp(design(p_new)))
    print(f"반복 {it + 1}: p = {p_new:.3f}, 예측 {np.polyval(coef, p_new):.5f} K, 재검증 {Js[-1]:.5f} K")
true_J = [peak_temp(design(p)) for p in grid]
print(f"참고: 이 설계군의 참 최적 p = {grid[np.argmin(true_J)]:.2f}, J = {min(true_J):.5f} K")
