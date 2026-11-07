import numpy as np


def ridge(X, y, lam):
    """ridge 회귀의 해 (X^T X + lam I) w = X^T y를 푼다.

    X: (N, D), y: (N,), lam: 양수
    반환: w (D,)
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    D = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(D), X.T @ y)   # (D,)
