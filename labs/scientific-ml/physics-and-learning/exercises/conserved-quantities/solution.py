import numpy as np


def conservation_laws(N, tol=1e-10):
    """화학량론 행렬 N (S, R)의 왼쪽 영공간 basis를 행으로 쌓은 W (K, S)를 반환한다.

    W의 행은 정규직교이고 W @ N ≈ 0이다. K = S - rank(N).
    """
    N = np.atleast_2d(np.asarray(N, dtype=float))
    U, s, _ = np.linalg.svd(N, full_matrices=True)       # U: (S, S), s: (min(S, R),)
    rank = int(np.sum(s > tol * s.max())) if s.size and s.max() > 0 else 0
    return U[:, rank:].T                                  # (S - rank, S)


def max_violation(Y, W):
    """궤적 Y (T, S)에서 max_t max_k |W (y_t - y_0)|를 반환한다."""
    Y, W = np.asarray(Y, dtype=float), np.asarray(W, dtype=float)
    if W.shape[0] == 0:
        return 0.0
    return float(np.abs((Y - Y[0]) @ W.T).max())          # (T, K)의 최댓값
