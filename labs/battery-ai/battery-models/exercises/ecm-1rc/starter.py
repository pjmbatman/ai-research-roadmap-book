import numpy as np


def simulate_1rc(current_a, dt_s, r0_ohm, r1_ohm, c1_f, q_ah, soc0, ocv):
    """1-RC 등가회로 모델을 영차 유지로 정확히 적분한다(방전 전류 +).

    current_a: (T,) [A], dt_s: 간격 [s], ocv: SOC -> OCV [V] 함수
    반환: voltage_v (T,) [V], soc (T,)
    """
    raise NotImplementedError("V1을 정확한 감쇠율 exp(-dt/tau)로 갱신하세요")
