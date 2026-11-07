import numpy as np


def rbf(a, b, ell, sf):
    """RBF 커널 행렬, (N,), (M,) → (N, M)"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return sf**2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)


def gp_posterior(x_train, y_train, x_test, ell, sf, sn):
    """잠재함수 f의 사후 (mean, var), 각각 shape (M,)"""
    raise NotImplementedError("K_y의 Cholesky 분해로 K_y^{-1} y와 분산을 계산하세요")


def log_marginal_likelihood(x_train, y_train, ell, sf, sn):
    """log p(y | X) = -y^T K_y^{-1} y / 2 - log det K_y / 2 - N log(2 pi) / 2"""
    raise NotImplementedError
