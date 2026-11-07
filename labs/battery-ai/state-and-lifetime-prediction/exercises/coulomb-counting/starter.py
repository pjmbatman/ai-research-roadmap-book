import numpy as np


def soc_coulomb(time_s, current_a, q_ah, soc0, eta_chg=1.0):
    """쿨롱 카운팅 SOC (방전 전류 +, 영차 유지, 충전 전류에만 eta_chg 적용).

    time_s, current_a: (T,) 배열. 반환: (T,) SOC, soc[0] = soc0. 0–1로 자르지 않는다.
    """
    raise NotImplementedError("구간마다 eta * I * dt / (3600 Q)를 빼 나가세요")


def offset_drift_pct(offset_a, hours, q_ah):
    """전류 센서 offset이 hours 시간 동안 쿨롱 카운팅 SOC에 쌓는 오차의 크기 [%p]."""
    raise NotImplementedError("offset이 쌓는 전하량을 용량으로 나누세요")
