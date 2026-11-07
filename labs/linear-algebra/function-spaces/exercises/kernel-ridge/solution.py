import numpy as np


def rbf(a, b, ell):
    """(n,), (m,) → (n, m) RBF 커널 행렬."""
    return np.exp(-(a[:, None] - b[None, :]) ** 2 / (2 * ell**2))


def kernel_ridge_predict(x_train, y_train, x_test, ell, lam):
    """RBF 커널 k(x, x') = exp(-(x - x')^2 / (2 ell^2))의 kernel ridge 예측.

    x_train (N,), y_train (N,), x_test (M,) → f(x_test) (M,)
    f(x) = sum_i alpha_i k(x, x_i),  (K + lam I) alpha = y
    """
    x_train, y_train, x_test = (np.asarray(v, dtype=float) for v in (x_train, y_train, x_test))
    K = rbf(x_train, x_train, ell)                                  # (N, N)
    alpha = np.linalg.solve(K + lam * np.eye(len(x_train)), y_train)  # (N,)
    return rbf(x_test, x_train, ell) @ alpha                         # (M,)
