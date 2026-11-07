"""30.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/battery-ai/battery-fundamentals/cells.py
"""
# %% cell 1 — C-rate에 따라 달라지는 방전 용량과 에너지
import numpy as np

Q_NOM_AH, V_NOM_V, R_DC_OHM, V_CUT_V = 3.0, 3.6, 0.035, 3.4   # 가상의 예시 셀

def ocv_v(soc):                                   # 가상의 예시 셀의 OCV-SOC 곡선 [V]
    return 3.5 + 0.55 * soc + 0.04 * np.log((soc + 0.02) / (1.02 - soc))

def discharge(c_rate, dt_s=1.0):
    i_a = c_rate * Q_NOM_AH                       # 1C = 공칭 용량을 1시간에 내보내는 전류
    t_s = np.arange(0, 3600 / c_rate, dt_s)       # (T,)
    soc = 1.0 - i_a * t_s / 3600 / Q_NOM_AH       # (T,) 정전류 방전
    v_v = ocv_v(soc) - i_a * R_DC_OHM             # (T,) 단자 전압 = OCV - IR 강하
    n = np.argmax(v_v < V_CUT_V) if np.any(v_v < V_CUT_V) else len(t_s)
    q_ah = i_a * n * dt_s / 3600                  # 컷오프까지 내보낸 전하량
    e_wh = np.sum(v_v[:n] * i_a) * dt_s / 3600    # 컷오프까지 내보낸 에너지
    return i_a, n * dt_s / 3600, q_ah, e_wh

print(f"공칭 에너지 = {Q_NOM_AH} Ah x {V_NOM_V} V = {Q_NOM_AH * V_NOM_V:.1f} Wh")
for c in [0.2, 1.0, 2.0, 3.0]:
    i_a, t_h, q_ah, e_wh = discharge(c)
    print(f"{c:3.1f}C: I = {i_a:3.1f} A, 시간 {t_h:4.2f} h, 용량 {q_ah:4.2f} Ah, "
          f"에너지 {e_wh:5.2f} Wh, 평균 전압 {e_wh / q_ah:4.2f} V")

# %% cell 2 — SOH의 두 정의와 등가 완전 사이클(EFC)
cycle = np.array([0, 200, 400, 600, 800])                 # RPT를 한 사이클 번호
cap_ah = np.array([3.08, 3.02, 2.93, 2.80, 2.58])          # RPT 용량 [Ah] (가상의 예시)
soh_nom = cap_ah / Q_NOM_AH                                # 공칭 용량 기준
soh_ini = cap_ah / cap_ah[0]                               # 첫 측정 용량 기준
for n, a, b in zip(cycle, soh_nom, soh_ini):
    print(f"cycle {n:3d}: SOH(공칭 기준) {a:6.1%}  SOH(초기 기준) {b:6.1%}")

dsoc = 0.80 - 0.30                                         # SOC 30% ↔ 80% 부분 사이클
throughput_ah = 800 * dsoc * Q_NOM_AH                      # 방전 방향 누적 전하량
print(f"부분 사이클 800회의 EFC = {throughput_ah:.0f} Ah / {Q_NOM_AH} Ah = {throughput_ah / Q_NOM_AH:.0f}")

# %% cell 3 — knee가 있는 용량 곡선과 선형 외삽
n = np.arange(0, 1501)                                           # 사이클
cap = 3.08 - 2.0e-4 * n - 0.02 * np.exp((n - 900) / 120)         # 가상의 예시 셀 [Ah]
eol_cap = 0.8 * Q_NOM_AH                                         # EOL: 공칭 용량의 80%
eol_true = n[np.argmax(cap < eol_cap)]
slope, icpt = np.polyfit(n[:400], cap[:400], 1)                  # 처음 400 사이클로 직선 맞춤
eol_lin = (eol_cap - icpt) / slope
print(f"EOL 기준 {eol_cap:.2f} Ah, 실제 EOL {eol_true} 사이클")
print(f"처음 400 사이클의 기울기 {slope * 1e3:.3f} mAh/사이클 → 선형 외삽 EOL {eol_lin:.0f} 사이클")
print(f"cycle 800·1000·1100 용량: {cap[800]:.3f}, {cap[1000]:.3f}, {cap[1100]:.3f} Ah")

# %% cell 4 — 전류 부호 규약 확인하기
def record(discharge_sign):                    # 휴지 → 2C 방전 → 휴지 → 1C 충전, 1 s 간격
    i_a = np.r_[np.zeros(60), np.full(600, 6.0), np.zeros(60), np.full(900, -3.0)]  # (T,) 방전 +
    soc = 0.9 - np.cumsum(i_a) / 3600 / Q_NOM_AH
    v_v = ocv_v(soc) - i_a * R_DC_OHM
    return discharge_sign * i_a, v_v           # 데이터셋이 쓰는 부호로 저장

for name, sign in [("데이터셋 A (방전 +)", +1), ("데이터셋 B (방전 -)", -1)]:
    i_a, v_v = record(sign)
    dvdt = np.gradient(v_v)                    # (T,) [V/s]
    pos = i_a > 0.1                            # 양의 전류가 흐르는 구간
    naive_ah = np.sum(i_a[pos]) / 3600         # "양의 전류 = 방전"이라 가정한 방전 용량
    print(f"{name}: I>0 구간의 평균 dV/dt {np.mean(dvdt[pos]) * 1e3:+.3f} mV/s, "
          f"I>0을 방전으로 본 용량 {naive_ah:.2f} Ah")
