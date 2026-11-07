import numpy as np


def conjugate_gradient(matvec, b, tol=1e-10, max_iter=1000):
    """SPD 시스템 A x = b를 CG로 푼다. A는 matvec(v) = A @ v로만 주어진다.

    b: (n,)
    반환: (x, n_iter) — x: (n,), n_iter: 수행한 반복 횟수
    """
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b)                  # (n,)
    r = b.copy()                          # r0 = b - A x0 = b
    p = r.copy()
    rr = r @ r
    b_norm = np.linalg.norm(b)
    for k in range(max_iter):
        if np.sqrt(rr) <= tol * b_norm:
            return x, k
        Ap = matvec(p)                    # 반복마다 한 번
        alpha = rr / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = r @ r
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x, max_iter
