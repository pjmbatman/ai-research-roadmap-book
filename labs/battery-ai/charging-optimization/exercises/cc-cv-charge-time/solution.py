import numpy as np


def cc_cv_time(i_cc_a, q_ah, r_ohm, ocv_a_v, ocv_b_v, v_max_v, soc0, i_cut_a):
    """내부 저항 모델(OCV = a + b·SOC, V = OCV + I·R)의 CC-CV 충전 시간.

    반환: (t_cc_s, t_cv_s) CC 단계와 CV 단계의 시간(초)
    """
    soc_cv = (v_max_v - ocv_a_v - i_cc_a * r_ohm) / ocv_b_v       # CC가 끝나는 SOC
    if soc0 < soc_cv:
        t_cc_s = (soc_cv - soc0) * 3600 * q_ah / i_cc_a
        i_start_a = i_cc_a
    else:                                                           # 처음부터 전압 상한: 바로 CV
        t_cc_s = 0.0
        i_start_a = (v_max_v - ocv_a_v - ocv_b_v * soc0) / r_ohm
    # CV: I = (V_max - a - b s)/R, ds/dt = I/(3600 Q) → dI/dt = -I · b / (3600 Q R)
    tau_s = 3600 * q_ah * r_ohm / ocv_b_v
    t_cv_s = tau_s * np.log(i_start_a / i_cut_a) if i_start_a > i_cut_a else 0.0
    return float(t_cc_s), float(t_cv_s)
