import numpy as np


def cc_cv_time(i_cc_a, q_ah, r_ohm, ocv_a_v, ocv_b_v, v_max_v, soc0, i_cut_a):
    """내부 저항 모델(OCV = a + b·SOC, V = OCV + I·R)의 CC-CV 충전 시간.

    반환: (t_cc_s, t_cv_s) CC 단계와 CV 단계의 시간(초)
    """
    raise NotImplementedError("CC가 끝나는 SOC와 CV 단계의 전류 감소를 계산하세요")
