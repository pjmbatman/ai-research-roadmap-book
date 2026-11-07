import numpy as np


def split_rhat(chains):
    """(M, N) 체인 배열의 split R-hat. 2차원이 아니거나 N < 4이면 ValueError."""
    c = np.asarray(chains, dtype=float)
    if c.ndim != 2 or c.shape[1] < 4:
        raise ValueError("chains는 shape (M, N), N >= 4인 배열이어야 합니다")
    h = c.shape[1] // 2
    c = np.vstack([c[:, :h], c[:, h:2 * h]])          # (2M, h)
    n = c.shape[1]
    W = c.var(axis=1, ddof=1).mean()                  # 체인 내 분산
    B = n * c.mean(axis=1).var(ddof=1)                # 체인 간 분산
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))
