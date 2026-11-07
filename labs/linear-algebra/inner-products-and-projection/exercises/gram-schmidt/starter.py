import numpy as np


def gram_schmidt(A):
    """열이 독립인 A를 Gram–Schmidt로 직교화해 A = Q R을 만든다.

    A: (m, n), n <= m
    반환: Q (m, n) 정규직교 열, R (n, n) 상삼각(대각 양수)
    """
    raise NotImplementedError("각 열에서 앞 방향으로의 사영을 빼고 정규화하세요")
