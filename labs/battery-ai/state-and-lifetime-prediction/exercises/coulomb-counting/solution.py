import numpy as np


def soc_coulomb(time_s, current_a, q_ah, soc0, eta_chg=1.0):
    """쿨롱 카운팅 SOC (방전 전류 +, 영차 유지, 충전 전류에만 eta_chg 적용).

    time_s, current_a: (T,) 배열. 반환: (T,) SOC, soc[0] = soc0. 0–1로 자르지 않는다.
    """
    t_s = np.asarray(time_s, dtype=float)
    i_a = np.asarray(current_a, dtype=float)
    dt_s = np.diff(t_s)                                   # (T-1,)
    i_k = i_a[:-1]
    eff = np.where(i_k < 0, eta_chg, 1.0)                 # 충전에만 효율
    dsoc = eff * i_k * dt_s / 3600 / q_ah
    return np.r_[soc0, soc0 - np.cumsum(dsoc)]


def offset_drift_pct(offset_a, hours, q_ah):
    """전류 센서 offset이 hours 시간 동안 쿨롱 카운팅 SOC에 쌓는 오차의 크기 [%p]."""
    return abs(offset_a) * hours / q_ah * 100
