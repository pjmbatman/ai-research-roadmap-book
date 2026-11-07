import numpy as np


def trend_test(y):
    """최소제곱 추세와, 잔차의 lag-1 자기상관으로 보정한 표준오차.

    y: (n,)
    반환: slope, se_naive, n_eff, se_adj
    """
    raise NotImplementedError("기울기, 표준오차, 유효 표본 수를 계산하세요")
