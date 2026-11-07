import numpy as np


def bilinear(x, W, z, c):
    """y[b, k] = sum_ij x[b, i] * W[k, i, j] * z[b, j] + c[k]

    x: (B, I), W: (K, I, J), z: (B, J), c: (K,)
    반환: (B, K)
    """
    x, W, z, c = map(np.asarray, (x, W, z, c))
    return np.einsum("bi,kij,bj->bk", x, W, z) + c   # (B, K) + (K,) → (B, K)
