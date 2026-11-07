import numpy as np


def damped_newton(F, J, x0, tol=1e-10, max_iter=50):
    """F(x) = 0을 backtracking line search가 있는 Newton 방법으로 푼다.

    F: (n,) → (n,), J: (n,) → (n, n), x0: (n,)
    반환: (x, n_iter)
    """
    x = np.asarray(x0, dtype=float).copy()
    for k in range(max_iter):
        fx = F(x)
        norm_f = np.linalg.norm(fx)
        if norm_f <= tol:
            return x, k
        dx = np.linalg.solve(J(x), -fx)           # Newton step
        t = 1.0
        while np.linalg.norm(F(x + t * dx)) >= norm_f and t >= 1e-8:
            t /= 2                                # backtracking
        x = x + t * dx
    return x, max_iter
