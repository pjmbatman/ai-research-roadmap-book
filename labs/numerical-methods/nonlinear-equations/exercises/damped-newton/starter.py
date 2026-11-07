import numpy as np


def damped_newton(F, J, x0, tol=1e-10, max_iter=50):
    """F(x) = 0을 backtracking line search가 있는 Newton 방법으로 푼다.

    F: (n,) → (n,), J: (n,) → (n, n), x0: (n,)
    반환: (x, n_iter)
    """
    raise NotImplementedError("J dx = -F를 풀고, |F|가 줄어들 때까지 t를 절반으로 줄이세요")
