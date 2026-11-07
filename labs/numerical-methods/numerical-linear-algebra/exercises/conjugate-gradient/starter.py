import numpy as np


def conjugate_gradient(matvec, b, tol=1e-10, max_iter=1000):
    """SPD 시스템 A x = b를 CG로 푼다. A는 matvec(v) = A @ v로만 주어진다.

    b: (n,)
    반환: (x, n_iter) — x: (n,), n_iter: 수행한 반복 횟수
    """
    raise NotImplementedError("x0 = 0, r0 = p0 = b에서 시작해 alpha, beta 갱신을 반복하세요")
