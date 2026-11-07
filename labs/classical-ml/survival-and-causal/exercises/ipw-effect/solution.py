import numpy as np


def naive_difference(y, t):
    """처치군 평균 - 대조군 평균. y: (N,), t: (N,) 0/1"""
    y, t = np.asarray(y, dtype=float), np.asarray(t).astype(bool)
    return float(y[t].mean() - y[~t].mean())


def propensity(Z, t, iters=25):
    """절편 포함 로지스틱 회귀(Newton)로 P(T=1 | Z)를 추정한다. Z: (N, p). 반환: (N,)"""
    Z = np.asarray(Z, dtype=float).reshape(len(t), -1)
    t = np.asarray(t, dtype=float)
    X = np.c_[np.ones(len(t)), Z]                                     # (N, p + 1)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        e = 1 / (1 + np.exp(-X @ w))
        w = w + np.linalg.solve(X.T @ (X * (e * (1 - e))[:, None]), X.T @ (t - e))
    return 1 / (1 + np.exp(-X @ w))


def ipw_effect(y, t, e):
    """정규화한 역확률 가중 추정량. e: 성향점수 (N,)"""
    y, e = np.asarray(y, dtype=float), np.asarray(e, dtype=float)
    t = np.asarray(t, dtype=float)
    w1, w0 = t / e, (1 - t) / (1 - e)
    return float(np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0))
