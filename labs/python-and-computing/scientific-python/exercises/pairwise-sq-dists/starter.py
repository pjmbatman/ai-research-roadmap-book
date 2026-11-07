import numpy as np


def pairwise_sq_dists(X, Y):
    """모든 쌍의 제곱 유클리드 거리 S[i, j] = ||X[i] - Y[j]||^2 를 반복문 없이 계산한다.

    X: (N, D), Y: (M, D)
    반환: (N, M), 모든 원소 >= 0
    """
    raise NotImplementedError("broadcasting이나 전개식으로 계산하세요")
