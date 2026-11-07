import numpy as np


def bilinear(x, W, z, c):
    """y[b, k] = sum_ij x[b, i] * W[k, i, j] * z[b, j] + c[k]

    x: (B, I), W: (K, I, J), z: (B, J), c: (K,)
    반환: (B, K)
    """
    raise NotImplementedError("einsum 문자열을 쓰세요")
