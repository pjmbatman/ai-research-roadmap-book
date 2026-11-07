"""30.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/battery-ai/battery-models/cells.py
"""
# %% cell 1 — 1-RC 등가회로 모델의 펄스 응답
import numpy as np

Q_AH, SOC0 = 3.0, 0.8                              # 가상의 예시 셀 (위젯과 같은 값)
R0_OHM, R1_OHM, C1_F = 0.020, 0.015, 2000.0        # tau = R1 C1 = 30 s

def ocv_v(soc):                                    # 가상의 예시 셀의 OCV-SOC 곡선 [V]
    return 3.5 + 0.55 * soc + 0.04 * np.log((soc + 0.02) / (1.02 - soc))

def simulate(i_a, dt_s, r0, rc, soc0=SOC0, euler=False):
    """i_a: (T,) 방전 + 전류. rc: [(R, C), ...] RC 쌍 목록. 반환: (T,) 단자 전압 [V]"""
    R, C = np.array(rc).T                          # (P,), (P,)
    a = np.exp(-dt_s / (R * C))                    # 영차 유지에서의 정확한 감쇠율
    v_rc, soc, v = np.zeros(len(R)), soc0, np.empty(len(i_a))
    for k, i in enumerate(i_a):
        v[k] = ocv_v(soc) - i * r0 - v_rc.sum()
        if euler:                                  # 전진 Euler: dV/dt = -V/(RC) + I/C
            v_rc = v_rc + dt_s * (-v_rc / (R * C) + i / C)
        else:
            v_rc = a * v_rc + R * (1 - a) * i
        soc -= i * dt_s / 3600 / Q_AH
    return v

t_s = np.arange(0, 901.0)                          # (T,) 1 s 간격
T_ON, T_OFF = 90, 450                              # 1C 방전 펄스 360 s
i_a = np.where((t_s >= T_ON) & (t_s < T_OFF), 3.0, 0.0)  # (T,)
v = simulate(i_a, 1.0, R0_OHM, [(R1_OHM, C1_F)])
for k in [89, 90, 120, 449, 450, 480, 900]:
    print(f"t = {k:3d} s  I = {i_a[k]:.1f} A  V = {v[k]:.4f} V")

# %% cell 2 — 정확한 이산화와 전진 Euler의 step 크기
ref = simulate(i_a, 1.0, R0_OHM, [(R1_OHM, C1_F)])                 # (901,) 기준해
for dt in [5.0, 30.0, 45.0, 90.0]:
    idx = np.arange(0, 901, int(dt))                              # dt 간격의 시각
    i_dt = i_a[idx]                                               # 펄스 경계가 dt의 배수
    for name, eu in [("정확", False), ("Euler", True)]:
        err = np.max(np.abs(simulate(i_dt, dt, R0_OHM, [(R1_OHM, C1_F)], euler=eu) - ref[idx]))
        print(f"dt = {dt:2.0f} s  dt/tau = {dt / 30:4.2f}  {name:5s} 최대 오차 {err * 1e3:9.3f} mV")

# %% cell 3 — 펄스 응답에서 파라미터 읽기
from scipy.optimize import curve_fit

rng = np.random.default_rng(0)
v_meas = v + rng.normal(0, 0.001, v.shape)                       # 전압 측정 잡음 1 mV
dv_on = v_meas[T_ON - 1] - v_meas[T_ON]                          # 펄스 시작의 즉시 강하
r0_hat = dv_on / 3.0
tr, vr = t_s[T_OFF + 1:] - T_OFF, v_meas[T_OFF + 1:]             # 휴지 구간 (펄스 직후 제외)
def relax(t, v_inf, a, tau):                                     # V(t) = V_inf - a e^{-t/tau}
    return v_inf - a * np.exp(-t / tau)
(v_inf, a_hat, tau_hat), _ = curve_fit(relax, tr, vr, p0=[vr[-1], 0.05, 10.0])
r1_hat = a_hat / (3.0 * (1 - np.exp(-(T_OFF - T_ON) / tau_hat)))  # 휴지 시작 시 V1 = I R1 (1 - e^{-360/tau})
print(f"R0: 참 {R0_OHM * 1e3:.1f} mOhm, 추정 {r0_hat * 1e3:.1f} mOhm")
print(f"tau: 참 {R1_OHM * C1_F:.1f} s, 추정 {tau_hat:.1f} s")
print(f"R1: 참 {R1_OHM * 1e3:.1f} mOhm, 추정 {r1_hat * 1e3:.1f} mOhm, C1 추정 {tau_hat / r1_hat:.0f} F")

# %% cell 4 — OCV 기울기와 전압으로 읽는 SOC의 민감도
def ocv_flat_v(soc):                     # 평탄 구간이 넓은 가상의 곡선 (LFP와 비슷한 모양)
    return 3.3 + 0.03 * soc + 0.04 * np.log((soc + 0.02) / (1.02 - soc))

for soc in [0.1, 0.3, 0.5, 0.7, 0.9]:
    h = 1e-4
    s1 = (ocv_v(soc + h) - ocv_v(soc - h)) / (2 * h) / 100       # [V / %SOC]
    s2 = (ocv_flat_v(soc + h) - ocv_flat_v(soc - h)) / (2 * h) / 100
    print(f"SOC {soc:.0%}: 기울기 {s1 * 1e3:5.2f} / {s2 * 1e3:5.2f} mV/%, "
          f"전압 오차 5 mV → SOC 오차 {0.005 / s1:4.1f} / {0.005 / s2:4.1f} %p")

# %% cell 5 — 적용 범위: 짧은 펄스로 맞춘 1-RC를 긴 방전에 쓰기
rc_true = [(0.015, 2000.0), (0.010, 60000.0)]                    # 참 셀: 2-RC (tau 30 s, 600 s)
t_fit = np.arange(0, 181.0)
i_fit = np.where((t_fit >= 10) & (t_fit < 70), 3.0, 0.0)         # 60 s 펄스 + 휴지
v_fit = simulate(i_fit, 1.0, R0_OHM, rc_true)
def model(_, r1, c1_kf):                                         # 1-RC, R0는 즉시 강하로 고정
    return simulate(i_fit, 1.0, R0_OHM, [(r1, c1_kf * 1e3)])
(r1_hat, c1_hat), _ = curve_fit(model, t_fit, v_fit, p0=[0.01, 1.0])
t_long = np.arange(0, 1801.0)
i_long = np.full(t_long.shape, 3.0)                              # 1C 연속 방전 30분
err = simulate(i_long, 1.0, R0_OHM, [(r1_hat, c1_hat * 1e3)]) - simulate(i_long, 1.0, R0_OHM, rc_true)
rmse_fit = np.sqrt(np.mean((model(None, r1_hat, c1_hat) - v_fit) ** 2))
print(f"맞춘 1-RC: R1 {r1_hat * 1e3:.1f} mOhm, tau {r1_hat * c1_hat * 1e3:.0f} s, 맞춤 구간 RMSE {rmse_fit * 1e3:.1f} mV")
for k in [60, 300, 900, 1800]:
    print(f"연속 방전 {k:4d} s: 1-RC 예측 - 참 = {err[k] * 1e3:+6.1f} mV")
