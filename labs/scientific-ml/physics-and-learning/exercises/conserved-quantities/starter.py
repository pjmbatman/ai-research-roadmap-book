import numpy as np


def conservation_laws(N, tol=1e-10):
    """화학량론 행렬 N (S, R)의 왼쪽 영공간 basis를 행으로 쌓은 W (K, S)를 반환한다.

    W의 행은 정규직교이고 W @ N ≈ 0이다. K = S - rank(N).
    """
    raise NotImplementedError("SVD로 왼쪽 영공간을 구하세요")


def max_violation(Y, W):
    """궤적 Y (T, S)에서 max_t max_k |W (y_t - y_0)|를 반환한다."""
    raise NotImplementedError("보존량의 최대 변화를 계산하세요")
