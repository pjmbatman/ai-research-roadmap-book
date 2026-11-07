import numpy as np


def tikhonov_svd(A, y, lam):
    """SVD 필터 인자로 Tikhonov 해 argmin ||A x - y||^2 + lam ||x||^2를 구한다.

    A: (m, n), y: (m,), lam > 0
    반환: (x (n,), f (min(m, n),)) — f는 필터 인자 s^2 / (s^2 + lam)
    """
    A, y = np.asarray(A, dtype=float), np.asarray(y, dtype=float)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)   # (m, r), (r,), (r, n)
    f = s**2 / (s**2 + lam)                            # (r,) 필터 인자
    x = Vt.T @ (s / (s**2 + lam) * (U.T @ y))          # f_i / s_i = s_i / (s_i^2 + lam): 0으로 나누지 않는다
    return x, f
