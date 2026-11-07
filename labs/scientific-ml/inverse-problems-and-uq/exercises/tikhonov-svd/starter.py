import numpy as np


def tikhonov_svd(A, y, lam):
    """SVD 필터 인자로 Tikhonov 해 argmin ||A x - y||^2 + lam ||x||^2를 구한다.

    A: (m, n), y: (m,), lam > 0
    반환: (x (n,), f (min(m, n),)) — f는 필터 인자 s^2 / (s^2 + lam)
    """
    raise NotImplementedError("SVD로 필터 인자와 해를 계산하세요")
