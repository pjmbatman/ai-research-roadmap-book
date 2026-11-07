import numpy as np


def gaussian_nll(beta, X, y, sigma):
    """y_i ~ N(x_i^T beta, sigma_i^2)의 평균 NLL [nat]. X: (N, p), y: (N,), sigma: (N,), beta: (p,)"""
    X, y, sigma, beta = map(np.asarray, (X, y, sigma, beta))
    r = y - X @ beta                                           # (N,)
    return float(np.mean(r**2 / (2 * sigma**2) + np.log(sigma) + 0.5 * np.log(2 * np.pi)))


def weighted_lstsq(X, y, sigma):
    """gaussian_nll을 최소화하는 beta (p,)를 반환한다."""
    X, y, sigma = map(np.asarray, (X, y, sigma))
    Xw, yw = X / sigma[:, None], y / sigma                      # 각 행을 sigma_i로 나눈다
    return np.linalg.lstsq(Xw, yw, rcond=None)[0]
