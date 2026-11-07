import numpy as np


def mmd2_unbiased(x, y, bw):
    """RBF kernel(대역폭 bw)로 계산한 불편 MMD² 추정값.

    x: (m, d), y: (n, d), bw > 0
    반환: float
    """
    x, y = np.atleast_2d(np.asarray(x, float)), np.atleast_2d(np.asarray(y, float))
    m, n = len(x), len(y)

    def k(a, b):                                             # (len(a), len(b))
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * bw**2))

    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    term_x = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))     # i ≠ i'
    term_y = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))     # j ≠ j'
    return float(term_x + term_y - 2 * kxy.mean())
