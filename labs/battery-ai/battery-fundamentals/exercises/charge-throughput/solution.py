import numpy as np


def throughput(time_s, current_a, voltage_v, q_nom_ah, discharge_sign=+1):
    """측정 기록에서 방전·충전 전하량 [Ah], 방전 에너지 [Wh], EFC를 계산한다.

    time_s, current_a, voltage_v: (T,) 배열. 영차 유지로 적분한다(마지막 측정은 기여하지 않음).
    discharge_sign: 방전 전류가 양수로 기록되었으면 +1, 음수면 -1.
    반환: {"discharge_ah", "charge_ah", "discharge_wh", "efc"}
    """
    t_s, v_v = np.asarray(time_s, float), np.asarray(voltage_v, float)
    i_a = discharge_sign * np.asarray(current_a, float)   # (T,) 방전 + 규약
    dt_h = np.diff(t_s) / 3600                            # (T-1,)
    i_k, v_k = i_a[:-1], v_v[:-1]                         # 영차 유지: 구간 시작 값
    dis = np.clip(i_k, 0, None)
    chg = np.clip(-i_k, 0, None)
    discharge_ah = float(np.sum(dis * dt_h))
    return {
        "discharge_ah": discharge_ah,
        "charge_ah": float(np.sum(chg * dt_h)),
        "discharge_wh": float(np.sum(dis * v_k * dt_h)),
        "efc": discharge_ah / q_nom_ah,
    }
