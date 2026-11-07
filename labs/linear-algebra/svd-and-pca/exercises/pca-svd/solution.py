import numpy as np


def pca(X, k):
    """중심화한 X의 SVD로 주성분 k개를 구한다.

    X: (N, D), k: 1 이상 D 이하
    반환: dict(mean (D,), components (k, D), explained_variance (k,), scores (N, k))
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)                                  # (D,)
    Xc = X - mean                                          # (N, D)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)      # (N, r), (r,), (r, D)
    return {
        "mean": mean,
        "components": Vt[:k],                              # (k, D)
        "explained_variance": s[:k] ** 2 / (len(X) - 1),   # (k,)
        "scores": U[:, :k] * s[:k],                        # (N, k) = Xc @ Vt[:k].T
    }
