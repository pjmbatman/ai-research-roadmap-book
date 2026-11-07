import numpy as np


def affine_flow_logpdf(Y, A, b):
    """y = A z + b, z ~ N(0, I_d)일 때 log p_Y(y).

    Y: (N, d), A: (d, d) 가역, b: (d,)
    반환: (N,)
    """
    raise NotImplementedError("z = A^{-1}(y - b)를 구하고 log p_Z(z) - log|det A|를 계산하세요")
