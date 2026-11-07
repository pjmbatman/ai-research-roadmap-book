import numpy as np


def posterior(m0, P0, A, R, y):
    """x ~ N(m0, P0), y = A x + eps, eps ~ N(0, R)일 때 p(x | y) = N(m, P).

    반환: (m (D,), P (D, D))
    """
    m0, P0 = np.asarray(m0, dtype=float), np.asarray(P0, dtype=float)
    A, R, y = np.asarray(A, dtype=float), np.asarray(R, dtype=float), np.asarray(y, dtype=float)
    S = A @ P0 @ A.T + R                                  # (M, M) y의 주변 공분산
    K = np.linalg.solve(S, A @ P0).T                      # (D, M) = P0 A^T S^{-1} (S, P0 대칭)
    m = m0 + K @ (y - A @ m0)
    P = P0 - K @ A @ P0
    return m, 0.5 * (P + P.T)
