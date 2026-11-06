import numpy as np


def gaussian_nll(y, mu, sigma):
    """평균 가우시안 NLL. sigma <= 0이 있으면 ValueError."""
    y, mu, sigma = (np.asarray(v, dtype=float) for v in (y, mu, sigma))
    if np.any(sigma <= 0):
        raise ValueError("sigma는 양수여야 합니다")
    nll = 0.5 * np.log(2 * np.pi * sigma**2) + (y - mu) ** 2 / (2 * sigma**2)
    return float(np.mean(nll))


def best_sigma(residuals, grid):
    """모든 샘플에 같은 sigma를 쓸 때 평균 NLL을 최소화하는 grid의 값."""
    r = np.asarray(residuals, dtype=float)
    losses = [gaussian_nll(r, 0.0, np.full_like(r, s)) for s in grid]
    return float(np.asarray(grid)[int(np.argmin(losses))])
