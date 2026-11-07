"""30.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/battery-ai/state-and-lifetime-prediction/cells.py
"""
# %% cell 1 — 쿨롱 카운팅의 오차 누적
import numpy as np

rng = np.random.default_rng(0)
DT_S, Q_TRUE_AH = 1.0, 2.7                                # 열화된 셀: 실제 용량 2.7 Ah (공칭 3.0)
seg = rng.uniform(-2.0, 3.6, 360)                         # 20 s마다 바뀌는 부하 [A], 방전 +
i_true = np.repeat(seg, 20)                               # (7200,) 2시간
soc_true = 0.9 - np.cumsum(i_true) * DT_S / 3600 / Q_TRUE_AH

def coulomb(i_meas, q_ah, soc0):
    return soc0 - np.cumsum(i_meas) * DT_S / 3600 / q_ah  # (T,) 영차 유지 적산

cases = {"초기 SOC 오차 +5%p": coulomb(i_true, Q_TRUE_AH, 0.95),
         "전류 센서 offset +0.05 A": coulomb(i_true + 0.05, Q_TRUE_AH, 0.9),
         "용량을 공칭 3.0 Ah로 가정": coulomb(i_true, 3.0, 0.9)}
print(f"실제 SOC: 0분 {soc_true[0]:.3f} → 120분 {soc_true[-1]:.3f}")
for name, soc in cases.items():
    e = (soc - soc_true) * 100                            # [%p]
    print(f"{name:22s} 오차 [%p]  30분 {e[1799]:+5.2f}  60분 {e[3599]:+5.2f}  120분 {e[-1]:+5.2f}")

# %% cell 2 — 확장 칼만 필터로 SOC 추정
R0, R1, C1 = 0.020, 0.015, 2000.0                         # 1-RC 파라미터 (가상의 예시 셀)
a1 = np.exp(-DT_S / (R1 * C1))
def ocv(s):
    return 3.5 + 0.55 * s + 0.04 * np.log((s + 0.02) / (1.02 - s))
def docv(s):
    return 0.55 + 0.04 * (1 / (s + 0.02) + 1 / (1.02 - s))

v1, v_meas = 0.0, np.empty_like(i_true)                   # 참 셀로 측정 전압 만들기
for k, i in enumerate(i_true):
    v_meas[k] = ocv(soc_true[k]) - i * R0 - v1 + rng.normal(0, 0.005)
    v1 = a1 * v1 + R1 * (1 - a1) * i
i_meas, q_model = i_true + 0.05, 3.0                      # 센서 offset, 공칭 용량 가정
x, P = np.array([0.6, 0.0]), np.diag([0.1**2, 0.01**2])   # 초기 추정 SOC 60% (참 90%)
F, Qn, Rn = np.diag([1.0, a1]), np.diag([1e-7, 1e-8]), 0.005**2
soc_ekf = np.empty_like(i_true)
for k, i in enumerate(i_meas):
    H = np.array([docv(x[0]), -1.0])                      # 관측 Jacobian dV/d(SOC, V1)
    S = H @ P @ H + Rn
    K = P @ H / S                                         # (2,) 칼만 이득
    x = x + K * (v_meas[k] - (ocv(x[0]) - i * R0 - x[1]))
    P = P - np.outer(K, H @ P)
    soc_ekf[k] = x[0]
    x = np.array([x[0] - i * DT_S / 3600 / q_model, a1 * x[1] + R1 * (1 - a1) * i])  # 예측
    P = F @ P @ F.T + Qn
e_cc = (coulomb(i_meas, q_model, 0.6) - soc_true) * 100
e_ekf = (soc_ekf - soc_true) * 100
for t in [0, 60, 600, 3599, 7199]:
    print(f"t = {t + 1:5d} s  쿨롱 카운팅 오차 {e_cc[t]:+6.2f} %p   EKF 오차 {e_ekf[t]:+6.2f} %p")

# %% cell 3 — 초기 사이클 특징으로 수명 예측: 셀 단위와 프로토콜 단위 검증
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict

n_proto, per = 8, 10
proto = np.repeat(np.arange(n_proto), per)                    # (80,) 충전 프로토콜
log_life = np.log(np.repeat(rng.uniform(400, 1600, n_proto), per)) + rng.normal(0, 0.15, 80)
life = np.exp(log_life)                                       # (80,) 수명 [사이클]
x_var = -1.0 * log_life + np.repeat(rng.normal(0, 0.15, n_proto), per) + rng.normal(0, 0.08, 80)
x_fade = 1e-4 * 100 * 900 / life + rng.normal(0, 0.006, 80)   # 사이클 10→100 용량 감소 [Ah]
designs = {"용량 감소": x_fade[:, None], "전압 곡선 특징": x_var[:, None]}
cvs = {"셀 단위 5-fold": list(KFold(5, shuffle=True, random_state=0).split(x_var)),
       "프로토콜 단위": list(LeaveOneGroupOut().split(x_var, groups=proto))}      # (학습, 테스트) 목록
print(f"평균값 예측 MAPE {np.mean(np.abs(np.exp(log_life.mean()) - life) / life):.1%}")
for fname, X in designs.items():
    row = []
    for cname, cv in cvs.items():
        p = np.exp(cross_val_predict(LinearRegression(), X, log_life, cv=cv))
        row.append(f"{cname} MAPE {np.mean(np.abs(p - life) / life):5.1%}")
    print(f"{fname:9s}: " + ",  ".join(row))

# %% cell 4 — split conformal 예측 구간과 조건별 coverage
def conformal_coverage(splits, alpha=0.1):
    hit = np.zeros(80, bool)
    for tr, te in splits:
        fit, cal = tr[::2], tr[1::2]                          # 학습 셀을 적합용·보정용으로 반씩
        m = LinearRegression().fit(x_var[fit, None], log_life[fit])
        r = np.sort(np.abs(log_life[cal] - m.predict(x_var[cal, None])))
        q = r[int(np.ceil((len(cal) + 1) * (1 - alpha))) - 1]  # 보정 잔차의 분위수
        hit[te] = np.abs(log_life[te] - m.predict(x_var[te, None])) <= q
    return hit

for cname, cv in cvs.items():
    hit = conformal_coverage(cv)
    by_proto = [hit[proto == g].mean() for g in range(n_proto)]
    print(f"{cname:12s} 전체 coverage {hit.mean():.0%}, 프로토콜별 최소 {min(by_proto):.0%} 최대 {max(by_proto):.0%}")
