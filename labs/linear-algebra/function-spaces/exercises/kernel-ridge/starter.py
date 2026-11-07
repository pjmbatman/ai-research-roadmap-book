import numpy as np


def kernel_ridge_predict(x_train, y_train, x_test, ell, lam):
    """RBF 커널 k(x, x') = exp(-(x - x')^2 / (2 ell^2))의 kernel ridge 예측.

    x_train (N,), y_train (N,), x_test (M,) → f(x_test) (M,)
    f(x) = sum_i alpha_i k(x, x_i),  (K + lam I) alpha = y
    """
    raise NotImplementedError("Gram 행렬을 만들고 alpha를 구하세요")
