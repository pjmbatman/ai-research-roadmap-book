import numpy as np


def leverage(X):
    """hat 행렬 X (XᵀX)⁻¹ Xᵀ의 대각 성분, shape (n,)."""
    X = np.asarray(X, dtype=float)
    return np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)   # 전체 (n, n) 행렬을 만들지 않는다


def cooks_distance(X, y):
    """각 관측의 Cook's distance, shape (n,)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    s2 = e @ e / (n - p)
    h = leverage(X)
    return e**2 / (p * s2) * h / (1 - h) ** 2
