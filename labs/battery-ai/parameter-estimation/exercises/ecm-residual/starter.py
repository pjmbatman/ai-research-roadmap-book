import numpy as np


def overpotential(r0_ohm, r1_ohm, c1_f, i_a, dt_s):
    """1-RC 등가회로 모델의 과전압 eta_k = I_k R0 + V1_k (V).

    i_a: (T,) 방전 전류(A, 방전이 양수), dt_s: 샘플 간격(s). 반환: (T,)
    """
    raise NotImplementedError("V1을 영차 유지로 갱신하며 과전압을 계산하세요")


def residual(log_theta, i_a, dt_s, eta_meas):
    """log_theta = (log R0, log R1, log C1)에서 모델 과전압 - 측정 과전압. 반환: (T,)"""
    raise NotImplementedError("log_theta를 되돌려 overpotential과 측정의 차이를 반환하세요")
