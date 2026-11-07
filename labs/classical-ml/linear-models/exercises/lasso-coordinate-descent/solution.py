import numpy as np


def soft_threshold(z, t):
    """S(z, t) = sign(z) * max(|z| - t, 0), 원소별."""
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def lasso_cd(X, y, alpha, n_sweeps=500):
    """(1/2N)||y - X w||² + alpha ||w||_1을 좌표 하강법으로 최소화한다.

    X: (N, p), 열마다 평균 0. y: (N,). 반환: w (p,)
    """
    X = np.asarray(X, dtype=float)                 # (N, p)
    y = np.asarray(y, dtype=float)                 # (N,)
    N, p = X.shape
    w = np.zeros(p)
    col_sq = (X**2).sum(axis=0) / N                # ||x_j||² / N, (p,)
    r = y - X @ w                                  # 잔차, (N,)
    for _ in range(n_sweeps):
        for j in range(p):
            rho = X[:, j] @ (r + X[:, j] * w[j]) / N
            w_new = soft_threshold(rho, alpha) / col_sq[j]
            r += X[:, j] * (w[j] - w_new)          # 잔차를 갱신
            w[j] = w_new
    return w
