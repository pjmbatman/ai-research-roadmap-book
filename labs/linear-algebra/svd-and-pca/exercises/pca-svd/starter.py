import numpy as np


def pca(X, k):
    """중심화한 X의 SVD로 주성분 k개를 구한다.

    X: (N, D), k: 1 이상 D 이하
    반환: dict(mean (D,), components (k, D), explained_variance (k,), scores (N, k))
    """
    raise NotImplementedError("평균을 빼고 SVD를 계산하세요")
