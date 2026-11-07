import numpy as np


def simulate_1rc(current_a, dt_s, r0_ohm, r1_ohm, c1_f, q_ah, soc0, ocv):
    """1-RC 등가회로 모델을 영차 유지로 정확히 적분한다(방전 전류 +).

    current_a: (T,) [A], dt_s: 간격 [s], ocv: SOC -> OCV [V] 함수
    반환: voltage_v (T,) [V], soc (T,)
    """
    i_a = np.asarray(current_a, dtype=float)
    a = np.exp(-dt_s / (r1_ohm * c1_f))           # 정확한 감쇠율
    v_v, soc = np.empty(len(i_a)), np.empty(len(i_a))
    v1, s = 0.0, soc0
    for k, i in enumerate(i_a):
        soc[k] = s
        v_v[k] = ocv(np.array(s)) - i * r0_ohm - v1
        v1 = a * v1 + r1_ohm * (1 - a) * i
        s = s - i * dt_s / 3600 / q_ah
    return v_v, soc
