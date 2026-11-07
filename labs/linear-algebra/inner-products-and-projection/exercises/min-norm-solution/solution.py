import numpy as np


def min_norm_solution(A, b):
    """full row rank인 미결정 시스템 A x = b의 최소 노름 해 x+ = A^T (A A^T)^{-1} b.

    A: (m, n), m < n, 행이 독립
    b: (m,)
    반환: x (n,)
    """
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    z = np.linalg.solve(A @ A.T, b)     # (m,)
    return A.T @ z                      # (n,) row space 안의 해
