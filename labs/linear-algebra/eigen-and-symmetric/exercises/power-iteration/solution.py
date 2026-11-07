import numpy as np


def power_iteration(A, num_iters=10000, tol=1e-10, seed=0):
    """거듭제곱법으로 지배 고윳값과 단위 고유벡터를 구한다.

    A: (n, n) 정사각행렬
    반환: lam (float, Rayleigh quotient), v (n,) 단위 벡터
    """
    A = np.asarray(A, dtype=float)
    v = np.random.default_rng(seed).normal(size=A.shape[0])   # (n,)
    v /= np.linalg.norm(v)
    lam = v @ A @ v
    for _ in range(num_iters):
        w = A @ v
        v = w / np.linalg.norm(w)
        lam_new = v @ A @ v                                    # Rayleigh quotient (|v| = 1)
        if abs(lam_new - lam) < tol:
            lam = lam_new
            break
        lam = lam_new
    return float(lam), v
