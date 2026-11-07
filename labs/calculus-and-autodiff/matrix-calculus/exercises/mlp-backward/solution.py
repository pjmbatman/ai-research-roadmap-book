import numpy as np


def mlp_grads(X, y, W1, b1, W2, b2):
    """2층 ReLU MLP + 평균 softmax cross-entropy의 파라미터 gradient.

    X: (N, D), y: (N,), W1: (H, D), b1: (H,), W2: (K, H), b2: (K,)
    반환: {"W1": (H, D), "b1": (H,), "W2": (K, H), "b2": (K,)}
    """
    X, W1, b1, W2, b2 = (np.asarray(a, dtype=float) for a in (X, W1, b1, W2, b2))
    N, K = X.shape[0], W2.shape[0]
    A = X @ W1.T + b1                              # (N, H)
    Hh = np.maximum(A, 0.0)                        # (N, H)
    Z = Hh @ W2.T + b2                             # (N, K)
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)   # (N, K)
    Y = np.eye(K)[np.asarray(y)]                   # (N, K)

    Gz = (P - Y) / N                               # (N, K)
    Ga = (Gz @ W2) * (A > 0)                       # (N, H)
    return {"W1": Ga.T @ X, "b1": Ga.sum(axis=0), "W2": Gz.T @ Hh, "b2": Gz.sum(axis=0)}
