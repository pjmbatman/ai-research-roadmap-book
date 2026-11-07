"""30.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/battery-ai/charging-optimization/cells.py
"""
# %% cell 1 — CC-CV 충전: 전류를 올리면 CV 구간이 길어진다
import numpy as np

Q_AH, R0_OHM, R1_OHM, C1_F = 3.0, 0.020, 0.015, 2000.0   # 30.5장의 가상 셀 (충전 전류를 양수로 둔다)
MCP_JPK, HA_WPK, T_AMB_C, V_MAX_V = 45.0, 0.10, 25.0, 4.2  # 열용량, 열전달 계수, 주변 온도, 상한 전압
def ocv_v(soc):
    return 3.5 + 0.55 * soc + 0.04 * np.log((soc + 0.02) / (1.02 - soc))

def cc_cv(c_rate, soc0=0.1, dt_s=1.0, i_cut_a=0.15):     # CC로 V_MAX까지, 이후 CV로 전류가 C/20이 될 때까지
    soc, v1, temp, t, cv = soc0, 0.0, T_AMB_C, 0.0, False
    a = np.exp(-dt_s / (R1_OHM * C1_F))
    t80, soc_cv, temp_max = None, None, temp
    while True:
        i = c_rate * Q_AH
        if cv or ocv_v(soc) + i * R0_OHM + v1 > V_MAX_V:   # 전압 상한에 닿으면 CV: 단자 전압 = V_MAX가 되게 전류를 줄인다
            cv, soc_cv = True, soc_cv or soc
            i = (V_MAX_V - ocv_v(soc) - v1) / R0_OHM
            if i < i_cut_a:
                return soc_cv, t80, t, temp_max
        temp += dt_s * (i * (i * R0_OHM + v1) - HA_WPK * (temp - T_AMB_C)) / MCP_JPK   # 줄 발열 - 냉각
        v1 = a * v1 + R1_OHM * (1 - a) * i
        soc += i * dt_s / (3600 * Q_AH); t += dt_s
        temp_max = max(temp_max, temp)
        if t80 is None and soc >= 0.8:
            t80 = t

for c in [0.5, 1.0, 2.0, 3.0]:
    soc_cv, t80, t_end, tmax = cc_cv(c)
    print(f"{c:3.1f}C  CV 시작 SOC {soc_cv:.2f}  80%까지 {t80/60:5.1f} min  종료까지 {t_end/60:5.1f} min  최고 {tmax:4.1f} °C")

# %% cell 2 — 2단계 CC 프로토콜의 다목적 trade-off와 Pareto front
def two_stage(c1, c2, soc0=0.1, soc_sw=0.5, soc_end=0.8, dt_s=2.0):   # c1, c2: (N,) 단계별 C-rate
    soc, v1, temp = np.full(c1.shape, soc0), np.zeros(c1.shape), np.full(c1.shape, T_AMB_C)
    t, wear, v_peak, t_peak = np.zeros(c1.shape), np.zeros(c1.shape), np.zeros(c1.shape), temp.copy()
    a = np.exp(-dt_s / (R1_OHM * C1_F))
    while np.any(soc < soc_end):
        i = np.where(soc < soc_end, np.where(soc < soc_sw, c1, c2) * Q_AH, 0.0)    # (N,) 끝난 후보는 0 A
        v_peak = np.maximum(v_peak, np.where(i > 0, ocv_v(soc) + i * R0_OHM + v1, 0.0))
        wear += (i / Q_AH) ** 2 * np.exp(3 * (soc - 0.5) + (temp - T_AMB_C) / 15) * dt_s / 3600  # 가상 열화 지표
        temp += dt_s * (i * (i * R0_OHM + v1) - HA_WPK * (temp - T_AMB_C)) / MCP_JPK
        v1 = a * v1 + R1_OHM * (1 - a) * i
        t += np.where(i > 0, dt_s, 0.0); soc += i * dt_s / (3600 * Q_AH)
        t_peak = np.maximum(t_peak, temp)
    return t / 60, wear, v_peak, t_peak                   # 충전 시간 (min), 열화 지표, 최고 전압 (V), 최고 온도 (°C)

grid = np.linspace(1.0, 4.0, 31)                           # 단계별 C-rate 후보
C1g, C2g = [g.ravel() for g in np.meshgrid(grid, grid, indexing="ij")]   # (961,)
t_min, wear, v_pk, temp_pk = two_stage(C1g, C2g)
feasible = (v_pk <= V_MAX_V) & (temp_pk <= 45.0)          # 안전 제약: 전압 4.2 V, 온도 45 °C

def pareto_mask(F):                                        # F: (N, 2) 최소화할 목적. 지배당하지 않는 점 True
    dominated = np.any(np.all(F[:, None, :] >= F[None, :, :], axis=2) & np.any(F[:, None, :] > F[None, :, :], axis=2), axis=1)
    return ~dominated

front = np.flatnonzero(feasible)[pareto_mask(np.column_stack([t_min, wear])[feasible])]
print(f"후보 {len(C1g)}개 중 제약을 만족하는 것 {feasible.sum()}개, 그중 Pareto 최적 {len(front)}개")
for k in front[np.argsort(t_min[front])][::10]:
    print(f"  {C1g[k]:.1f}C → {C2g[k]:.1f}C  시간 {t_min[k]:5.1f} min  열화 {wear[k]:.3f}  최고 {v_pk[k]:.3f} V, {temp_pk[k]:4.1f} °C")
same = np.flatnonzero(np.isclose(C1g, C2g) & feasible)
print("1단계(같은 C-rate) 중 Pareto 최적:", int(np.isin(same, front).sum()), "/", len(same))

# %% cell 3 — 제약이 있는 베이지안 최적화: GP 두 개와 EI × 실현 가능 확률
from scipy.stats import norm

X_all = np.column_stack([C1g, C2g]) / 4.0                   # (961, 2) 입력을 [0.25, 1]로
g_all = np.maximum.reduce([(v_pk - V_MAX_V) / 0.05, (temp_pk - 45.0) / 5.0, (t_min - 20.0) / 2.0])  # ≤ 0이면 제약 만족
f_all = np.log(wear)                                         # 목적: 가상 열화 지표(로그)를 최소화
def gp(X, y, Xs, ell=0.15, noise=1e-6):                      # RBF 커널 GP의 사후 평균·표준편차 (y 표준화)
    k = lambda A, B: np.exp(-0.5 * np.sum((A[:, None] - B[None]) ** 2, -1) / ell**2)
    m, s = y.mean(), y.std() + 1e-12
    L = np.linalg.cholesky(k(X, X) + noise * np.eye(len(X)))
    v = np.linalg.solve(L, k(X, Xs))                         # (n, C)
    return m + s * (v.T @ np.linalg.solve(L, (y - m) / s)), s * np.sqrt(np.clip(1 - np.sum(v**2, 0), 1e-12, None))

def next_point(idx):                                         # idx: 지금까지 평가한 후보 번호
    X, ok = X_all[idx], g_all[idx] <= 0
    mu_g, sd_g = gp(X, g_all[idx], X_all)
    pof = norm.cdf(-mu_g / sd_g)                             # 제약을 만족할 확률
    if not ok.any():
        score = pof                                          # 실현 가능한 점을 아직 못 찾았으면 그것부터
    else:
        mu, sd = gp(X, f_all[idx], X_all)
        z = (f_all[idx][ok].min() - mu) / sd
        score = sd * (z * norm.cdf(z) + norm.pdf(z)) * pof   # EI × 실현 가능 확률
    score[idx] = -np.inf                                     # 이미 평가한 점 제외
    return int(np.argmax(score))

# %% cell 4 — 같은 평가 예산에서 BO와 random search
best_true = f_all[g_all <= 0].min()
def run(method, seed, budget=20):
    rng = np.random.default_rng(seed)
    idx = list(rng.choice(len(X_all), 4, replace=False))     # 처음 4개는 무작위
    while len(idx) < budget:
        idx.append(next_point(np.array(idx)) if method == "BO" else
                   int(rng.choice(np.setdiff1d(np.arange(len(X_all)), idx))))
    f = np.where(g_all[idx] <= 0, f_all[idx], np.inf)
    return np.minimum.accumulate(f) - best_true              # 평가 횟수별 regret (실현 가능한 점이 없으면 inf)

for method in ["BO", "random"]:
    R = np.array([run(method, s) for s in range(20)])        # (시드 20, 예산 20)
    for n in [10, 20]:
        print(f"{method:6s} 평가 {n}번: 실현 가능한 점을 찾은 비율 {np.mean(np.isfinite(R[:, n - 1])):.2f}, "
              f"최적의 5% 이내에 도달한 비율 {np.mean(R[:, n - 1] < np.log(1.05)):.2f}")
k = np.flatnonzero(f_all == best_true)[0]
print(f"참 최적: {C1g[k]:.1f}C → {C2g[k]:.1f}C, 시간 {t_min[k]:.1f} min, 열화 {wear[k]:.3f}, 최고 {temp_pk[k]:.1f} °C")
