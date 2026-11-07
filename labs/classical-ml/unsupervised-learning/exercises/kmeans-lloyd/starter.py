import numpy as np


def kmeans(X, init_centers, n_iter=100):
    """Lloyd 반복으로 k-means를 푼다.

    X: (N, D), init_centers: (K, D)
    반환: centers (K, D), labels (N,), history (반복마다 배정 직후의 inertia 리스트)
    """
    raise NotImplementedError("배정과 갱신을 번갈아 반복하세요")
