import numpy as np


def soft_threshold(v, t):
    """t * ||.||_1의 근위 연산: 성분마다 sign(v) * max(|v| - t, 0)."""
    v = np.asarray(v, dtype=float)
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def ista(A, y, lam, steps=2000):
    """0.5 ||A w - y||^2 + lam ||w||_1 을 proximal gradient로 최소화한다.

    A: (N, D), y: (N,), lam: 0 이상
    반환: w (D,)
    """
    A, y = np.asarray(A, dtype=float), np.asarray(y, dtype=float)
    eta = 1.0 / np.linalg.norm(A, 2) ** 2       # 1 / L
    w = np.zeros(A.shape[1])                    # (D,)
    for _ in range(steps):
        w = soft_threshold(w - eta * A.T @ (A @ w - y), eta * lam)
    return w
