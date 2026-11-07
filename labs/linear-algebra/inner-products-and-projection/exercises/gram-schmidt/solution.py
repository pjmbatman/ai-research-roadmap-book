import numpy as np


def gram_schmidt(A):
    """열이 독립인 A를 Gram–Schmidt로 직교화해 A = Q R을 만든다.

    A: (m, n), n <= m
    반환: Q (m, n) 정규직교 열, R (n, n) 상삼각(대각 양수)
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))                       # (m, n)
    R = np.zeros((n, n))                       # (n, n)
    for k in range(n):
        v = A[:, k].copy()                     # (m,)
        for _ in range(2):                     # modified GS를 두 번 반복(재직교화)
            for j in range(k):
                c = Q[:, j] @ v
                R[j, k] += c
                v -= c * Q[:, j]
        R[k, k] = np.linalg.norm(v)
        Q[:, k] = v / R[k, k]
    return Q, R
