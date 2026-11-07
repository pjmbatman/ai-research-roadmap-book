import numpy as np


def interval_score(lower, upper, y, alpha):
    """중심 (1 - alpha) 예측 구간 [lower, upper]의 평균 interval score."""
    l, u, y = (np.asarray(a, dtype=float) for a in (lower, upper, y))
    below = (l - y) * (y < l)          # 아래로 벗어난 거리 (안이면 0)
    above = (y - u) * (y > u)          # 위로 벗어난 거리 (안이면 0)
    return float(np.mean((u - l) + 2.0 / alpha * (below + above)))


def best_symmetric_halfwidth(y, center, alpha, grid):
    """구간 [center - h, center + h]의 평균 interval score를 가장 작게 만드는 h (grid 중에서)."""
    scores = [interval_score(center - h, center + h, y, alpha) for h in grid]
    return float(np.asarray(grid)[int(np.argmin(scores))])
