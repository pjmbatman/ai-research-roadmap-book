import numpy as np


def cohen_kappa(a, b):
    """두 평가자의 범주 판정 a, b (길이 N)의 Cohen kappa를 반환한다.

    kappa = (p_o - p_e) / (1 - p_e)
    p_o: 관측 일치율, p_e: 두 평가자의 범주 비율로 계산한 우연 일치율.
    두 평가자가 모두 같은 한 범주만 썼으면 float("nan"), 길이가 다르면 ValueError.
    """
    raise NotImplementedError("p_o와 p_e를 계산해 kappa를 반환하세요")
