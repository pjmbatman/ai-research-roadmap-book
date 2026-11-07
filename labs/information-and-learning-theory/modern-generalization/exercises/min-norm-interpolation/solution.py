import numpy as np


def min_norm_solution(X, y):
    """X w = y를 만족하는 해 중 노름이 가장 작은 해 (p,). pinv를 쓰지 않는다."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    return X.T @ np.linalg.solve(X @ X.T, y)                 # (p,)


def gd_limit(X, y, w0):
    """0.5‖Xw - y‖²에 w0에서 시작한 경사하강이 수렴하는 해 (p,)를 닫힌 형태로 계산한다."""
    X, w0 = np.asarray(X, dtype=float), np.asarray(w0, dtype=float)
    return w0 + min_norm_solution(X, np.asarray(y) - X @ w0)  # w0 + 행 공간 성분
