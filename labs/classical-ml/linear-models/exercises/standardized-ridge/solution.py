import numpy as np


def fit_ridge(X, y, lam):
    """표준화한 특징에 Ridge 벌점을 걸고, 원래 단위의 (w, b)를 반환한다.

    X: (N, p), y: (N,), lam: 정규화 강도 λ ≥ 0
    반환: w (p,), b (스칼라) — 예측은 X @ w + b
    """
    X = np.asarray(X, dtype=float)                  # (N, p)
    y = np.asarray(y, dtype=float)                  # (N,)
    mu, s = X.mean(axis=0), X.std(axis=0)           # (p,), (p,)
    Z = (X - mu) / s                                # (N, p)
    y_bar = y.mean()
    v = np.linalg.solve(Z.T @ Z + lam * np.eye(X.shape[1]), Z.T @ (y - y_bar))   # (p,)
    w = v / s                                       # 원래 단위의 계수, (p,)
    b = y_bar - mu @ w
    return w, b
