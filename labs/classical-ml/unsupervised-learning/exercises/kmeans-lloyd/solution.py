import numpy as np


def kmeans(X, init_centers, n_iter=100):
    """Lloyd 반복으로 k-means를 푼다.

    X: (N, D), init_centers: (K, D)
    반환: centers (K, D), labels (N,), history (반복마다 배정 직후의 inertia 리스트)
    """
    X = np.asarray(X, dtype=float)
    C = np.array(init_centers, dtype=float)                        # (K, D) 복사본
    labels, history = None, []
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)        # (N, K)
        new = d2.argmin(axis=1)                                    # (N,) 배정
        history.append(float(d2[np.arange(len(X)), new].sum()))
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for j in range(len(C)):                                    # 갱신
            if np.any(labels == j):
                C[j] = X[labels == j].mean(axis=0)
    labels = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(axis=1)   # 마지막 중심에 대한 배정
    return C, labels, history
