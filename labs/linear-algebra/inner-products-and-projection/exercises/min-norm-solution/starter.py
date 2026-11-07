import numpy as np


def min_norm_solution(A, b):
    """full row rank인 미결정 시스템 A x = b의 최소 노름 해 x+ = A^T (A A^T)^{-1} b.

    A: (m, n), m < n, 행이 독립
    b: (m,)
    반환: x (n,)
    """
    raise NotImplementedError("A A^T z = b를 풀고 x = A^T z를 반환하세요")
