import numpy as np


def stationary_distribution(P):
    """행 확률행렬 P (K, K)의 정상분포 pi (K,)를 구한다. pi @ P = pi, pi.sum() = 1."""
    P = np.asarray(P, dtype=float)
    K = P.shape[0]
    A = np.vstack([P.T - np.eye(K), np.ones((1, K))])   # (K + 1, K)
    b = np.zeros(K + 1)
    b[-1] = 1.0                                          # 마지막 줄: 합이 1
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)           # (K,)
    pi = np.clip(pi, 0.0, None)                          # 반올림 오차로 생긴 아주 작은 음수 제거
    return pi / pi.sum()
