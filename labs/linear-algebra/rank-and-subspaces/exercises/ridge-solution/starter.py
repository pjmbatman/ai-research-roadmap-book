import numpy as np


def ridge(X, y, lam):
    """ridge 회귀의 해 (X^T X + lam I) w = X^T y를 푼다.

    X: (N, D), y: (N,), lam: 양수
    반환: w (D,)
    """
    raise NotImplementedError("정규방정식에 lam * I를 더해 푸세요")
