"""30.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/battery-ai/parameter-estimation/cells.py
"""
# %% cell 1 — 1-RC 등가회로 모델을 펄스 응답에 맞추기
import numpy as np
from scipy.optimize import least_squares

Q_AH = 3.0                                           # 용량 (설명용 값)
def ocv_v(soc):                                      # 개방 전압 곡선 (설명용, 저율 시험으로 따로 쟀다고 가정)
    return 3.5 + 0.55 * soc + 0.04 * np.log((soc + 0.02) / (1.02 - soc))

def simulate_v(theta, t_s, i_a, soc0=0.8):           # 1-RC ECM, I > 0이 방전. theta = (R0 [Ω], R1 [Ω], C1 [F])
    r0_ohm, r1_ohm, c1_f = theta                     # 스칼라 또는 (G,) 배열(격자 계산용)
    dt_s = t_s[1] - t_s[0]
    a = np.exp(-dt_s / (r1_ohm * c1_f))              # step 동안 전류가 일정하면(영차 유지) 정확한 갱신
    soc = soc0 - np.concatenate([[0], np.cumsum(i_a[:-1])]) * dt_s / (3600 * Q_AH)   # (T,)
    v1, v1_hist = 0.0 * a, []
    for i_k in i_a:
        v1_hist.append(v1)
        v1 = a * v1 + r1_ohm * (1 - a) * i_k
    return ocv_v(soc) - i_a * r0_ohm - np.stack(v1_hist, axis=-1)   # (T,) 또는 (G, T)

t_s = np.arange(0, 600.0)                            # (600,) 1 Hz 샘플
i_a = np.where((t_s >= 60) & (t_s < 360), 3.0, 0.0)  # 3 A 방전 펄스 300 s, 앞뒤 휴지
theta_true = np.array([0.020, 0.015, 2000.0])        # R0 20 mΩ, R1 15 mΩ, C1 2 kF → τ = 30 s
rng = np.random.default_rng(0)
v_meas = simulate_v(theta_true, t_s, i_a) + rng.normal(0, 1e-3, t_s.shape)   # 측정 잡음 1 mV

def residual(log_theta):                             # 로그 파라미터: 양수 보장, 척도가 다른 파라미터를 상대 변화로
    return simulate_v(np.exp(log_theta), t_s, i_a) - v_meas
fit = least_squares(residual, np.log([0.01, 0.01, 1000.0]))
r0, r1, c1 = np.exp(fit.x)
print(f"추정: R0 = {r0*1e3:.2f} mΩ, R1 = {r1*1e3:.2f} mΩ, C1 = {c1:.0f} F, τ = {r1*c1:.1f} s")
print(f"잔차 RMS = {np.sqrt(np.mean(fit.fun**2))*1e3:.2f} mV (측정 잡음 1 mV)")

# %% cell 2 — 실험 설계에 따른 식별 가능성
def sensitivity(t_s, i_a, idx):                      # 참값에서 로그 파라미터 민감도 행렬 (len(idx), 3)
    f = lambda lt: simulate_v(np.exp(lt), t_s, i_a)[idx]
    lt0, h = np.log(theta_true), 1e-6
    return np.column_stack([(f(lt0 + h * e) - f(lt0 - h * e)) / (2 * h) for e in np.eye(3)])

t_short = np.arange(0, 24.0)                         # 펄스 3 s만, 휴지 없음
i_short = np.where(t_short >= 20, 3.0, 0.0)
designs = {"1 Hz, 펄스 300 s + 휴지": (t_s, i_a, np.arange(600)),
           "같은 실험, 60 s 간격 샘플": (t_s, i_a, np.arange(30, 600, 60)),
           "1 Hz, 펄스 3 s, 휴지 없음": (t_short, i_short, np.arange(24))}
for name, (tt, ii, idx) in designs.items():
    S = sensitivity(tt, ii, idx)
    cov = 1e-3**2 * np.linalg.inv(S.T @ S)           # 잡음 1 mV에서의 근사 공분산 (로그 파라미터)
    sd = np.sqrt(np.diag(cov)); corr = cov / np.outer(sd, sd)
    print(f"{name:18s} 상대오차 R0 {sd[0]:6.3f} R1 {sd[1]:6.3f} C1 {sd[2]:6.3f}"
          f" | 상관 R0–R1 {corr[0, 1]:+.3f}, R1–C1 {corr[1, 2]:+.3f}")

# %% cell 3 — 베이지안 추정: 사후분포가 사전분포와 같아지는 경우
lr1 = np.linspace(np.log(2e-3), np.log(0.1), 400)     # log R1 격자
lc1 = np.linspace(np.log(200.0), np.log(2e4), 150)    # log C1 격자
LR, LC = [g.ravel() for g in np.meshgrid(lr1, lc1, indexing="ij")]   # (G,) G = 400 × 150
log_prior = -0.5 * ((LR - np.log(0.010)) / 0.5) ** 2 - 0.5 * ((LC - np.log(1500.0)) / 0.5) ** 2  # 사전: 같은 모델 셀들의 값

def r1_summary(log_w):                               # 격자 가중치 → R1 주변분포의 중앙값과 90% 구간 [mΩ]
    w = np.exp(log_w - log_w.max()).reshape(400, 150).sum(axis=1)
    cdf = np.cumsum(w) / w.sum()
    return [np.exp(np.interp(q, cdf, lr1)) * 1e3 for q in (0.05, 0.5, 0.95)]

print("사전분포   R1 = {1:5.1f} mΩ  [{0:5.1f}, {2:5.1f}]".format(*r1_summary(log_prior)))
data = {"펄스 300 s, 60 s 간격": (t_s[::30], i_a[::30], slice(1, None, 2)),   # 30 s step으로 계산, t = 30, 90, ... s만 기록
        "펄스 3 s, 휴지 없음  ": (t_short, i_short, slice(None))}
for name, (tt, ii, keep) in data.items():
    y = (simulate_v(theta_true, tt, ii) + rng.normal(0, 1e-3, tt.shape))[keep]
    pred = simulate_v((theta_true[0], np.exp(LR), np.exp(LC)), tt, ii)[:, keep]   # (G, T), R0는 알려졌다고 고정
    log_post = log_prior - 0.5 * np.sum((pred - y) ** 2, axis=1) / 1e-3**2
    print(f"사후({name}) R1 = {{1:5.1f}} mΩ  [{{0:5.1f}}, {{2:5.1f}}]".format(*r1_summary(log_post)))

# %% cell 4 — 물리 구조가 있는 모델과 없는 모델의 외삽
T_meas_c = np.array([15.0, 20.0, 25.0, 30.0, 35.0])  # 측정 온도 (°C)
def r0_true(T_c):                                    # 가상 셀의 참 R0(T): Arrhenius, Ea = 30 kJ/mol (설명용)
    return 0.020 * np.exp(30e3 / 8.314 * (1 / (T_c + 273.15) - 1 / 298.15))
r0_meas = r0_true(T_meas_c) * (1 + 0.02 * np.random.default_rng(1).normal(size=5))   # 추정 오차 2%

poly = np.polyfit(T_meas_c, r0_meas, 2)              # 데이터 기반: T의 2차 다항식
arr = np.polyfit(1 / (T_meas_c + 273.15), np.log(r0_meas), 1)   # 물리 구조: ln R0 = a + (Ea/R)(1/T)
for T_c in [-10.0, 0.0, 25.0, 45.0]:
    p, a = np.polyval(poly, T_c), np.exp(np.polyval(arr, 1 / (T_c + 273.15)))
    print(f"T = {T_c:5.1f} °C  참값 {r0_true(T_c)*1e3:5.1f} mΩ | 2차 다항식 {p*1e3:5.1f} | Arrhenius {a*1e3:5.1f}")
print(f"Arrhenius 적합의 활성화 에너지 = {arr[0] * 8.314 / 1e3:.1f} kJ/mol (참값 30)")
