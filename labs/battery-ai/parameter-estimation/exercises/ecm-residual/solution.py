import numpy as np


def overpotential(r0_ohm, r1_ohm, c1_f, i_a, dt_s):
    """1-RC 등가회로 모델의 과전압 eta_k = I_k R0 + V1_k (V).

    i_a: (T,) 방전 전류(A, 방전이 양수), dt_s: 샘플 간격(s). 반환: (T,)
    """
    i_a = np.asarray(i_a, dtype=float)
    a = np.exp(-dt_s / (r1_ohm * c1_f))
    v1_v = np.zeros_like(i_a)                 # (T,) V1_k, V1_0 = 0
    for k in range(len(i_a) - 1):
        v1_v[k + 1] = a * v1_v[k] + r1_ohm * (1 - a) * i_a[k]
    return i_a * r0_ohm + v1_v


def residual(log_theta, i_a, dt_s, eta_meas):
    """log_theta = (log R0, log R1, log C1)에서 모델 과전압 - 측정 과전압. 반환: (T,)"""
    r0_ohm, r1_ohm, c1_f = np.exp(log_theta)
    return overpotential(r0_ohm, r1_ohm, c1_f, i_a, dt_s) - np.asarray(eta_meas)
