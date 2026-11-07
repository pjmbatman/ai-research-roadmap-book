import numpy as np


def rbf(a, b, ell, sf):
    """RBF 커널 행렬, (N,), (M,) → (N, M)"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return sf**2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)


def _chol(x_train, ell, sf, sn):
    x = np.asarray(x_train, dtype=float)
    return np.linalg.cholesky(rbf(x, x, ell, sf) + sn**2 * np.eye(x.size))   # (N, N)


def gp_posterior(x_train, y_train, x_test, ell, sf, sn):
    """잠재함수 f의 사후 (mean, var), 각각 shape (M,)"""
    L = _chol(x_train, ell, sf, sn)
    y = np.asarray(y_train, dtype=float)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))           # (N,) K_y^{-1} y
    Ks = rbf(x_train, x_test, ell, sf)                            # (N, M)
    v = np.linalg.solve(L, Ks)                                    # (N, M)
    return Ks.T @ alpha, sf**2 - np.sum(v**2, axis=0)


def log_marginal_likelihood(x_train, y_train, ell, sf, sn):
    """log p(y | X) = -y^T K_y^{-1} y / 2 - log det K_y / 2 - N log(2 pi) / 2"""
    L = _chol(x_train, ell, sf, sn)
    y = np.asarray(y_train, dtype=float)
    z = np.linalg.solve(L, y)                                     # y^T K_y^{-1} y = |z|^2
    return float(-0.5 * z @ z - np.sum(np.log(np.diag(L))) - 0.5 * y.size * np.log(2 * np.pi))
