import numpy as np


def pairwise_sq_dists(X, Y):
    """모든 쌍의 제곱 유클리드 거리 S[i, j] = ||X[i] - Y[j]||^2 를 반복문 없이 계산한다.

    X: (N, D), Y: (M, D)
    반환: (N, M), 모든 원소 >= 0
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    x2 = np.sum(X**2, axis=1)[:, None]       # (N, 1)
    y2 = np.sum(Y**2, axis=1)[None, :]       # (1, M)
    S = x2 - 2.0 * (X @ Y.T) + y2            # (N, M)
    return np.maximum(S, 0.0)                # 반올림 오차로 생긴 음수 제거
