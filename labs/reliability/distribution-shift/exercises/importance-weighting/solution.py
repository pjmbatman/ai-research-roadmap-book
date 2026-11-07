import numpy as np


def weighted_risk(losses, w):
    """자기 정규화 중요도 가중 평균 손실 sum(w * loss) / sum(w)."""
    losses, w = np.asarray(losses, dtype=float), np.asarray(w, dtype=float)
    return float(np.sum(w * losses) / np.sum(w))


def effective_sample_size(w):
    """유효 표본 수 (sum w)^2 / sum(w^2)."""
    w = np.asarray(w, dtype=float)
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def weighted_line_fit(x, y, w):
    """sum_i w_i (y_i - a - b x_i)^2 를 최소화하는 (a, b)."""
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))
    A = np.stack([np.ones_like(x), x], axis=1)                 # (N, 2)
    a, b = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))
    return float(a), float(b)
