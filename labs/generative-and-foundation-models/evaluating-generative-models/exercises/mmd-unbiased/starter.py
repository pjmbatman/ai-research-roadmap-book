import numpy as np


def mmd2_unbiased(x, y, bw):
    """RBF kernel(대역폭 bw)로 계산한 불편 MMD² 추정값.

    x: (m, d), y: (n, d), bw > 0
    반환: float
    """
    raise NotImplementedError("불편 MMD²를 계산하세요")
