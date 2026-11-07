import numpy as np


def low_rank_approx(A, k):
    """truncated SVD로 최적 rank-k 근사와 그 Frobenius 오차를 구한다.

    A: (m, n), k: 0 이상의 정수
    반환: A_k (m, n), err (float) = sqrt(버린 특이값 제곱합)
    """
    raise NotImplementedError("SVD의 앞 k개 항을 남기세요")
