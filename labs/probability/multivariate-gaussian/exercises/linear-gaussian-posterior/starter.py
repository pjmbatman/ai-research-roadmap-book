import numpy as np


def posterior(m0, P0, A, R, y):
    """x ~ N(m0, P0), y = A x + eps, eps ~ N(0, R)일 때 p(x | y) = N(m, P).

    반환: (m (D,), P (D, D))
    """
    raise NotImplementedError("gain 형태 또는 정밀도 형태의 공식을 구현한다")
