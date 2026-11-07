import numpy as np


def throughput(time_s, current_a, voltage_v, q_nom_ah, discharge_sign=+1):
    """측정 기록에서 방전·충전 전하량 [Ah], 방전 에너지 [Wh], EFC를 계산한다.

    time_s, current_a, voltage_v: (T,) 배열. 영차 유지로 적분한다(마지막 측정은 기여하지 않음).
    discharge_sign: 방전 전류가 양수로 기록되었으면 +1, 음수면 -1.
    반환: {"discharge_ah", "charge_ah", "discharge_wh", "efc"}
    """
    raise NotImplementedError("전류를 방전 + 규약으로 바꾸고 방향별로 적분하세요")
