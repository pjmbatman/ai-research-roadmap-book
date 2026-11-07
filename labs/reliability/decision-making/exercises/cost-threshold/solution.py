import numpy as np


def bayes_threshold(c_fp, c_fn):
    """보정된 확률에서 기대 비용을 최소화하는 경보 threshold."""
    return c_fp / (c_fp + c_fn)


def mean_cost(p, y, thr, c_fp, c_fn):
    """p > thr이면 경보. y = 1 고장, 0 정상. 사례당 평균 비용."""
    p, y = np.asarray(p, dtype=float), np.asarray(y).astype(bool)
    alarm = p > thr
    return float(np.mean(c_fp * (alarm & ~y) + c_fn * (~alarm & y)))


def best_threshold(p, y, c_fp, c_fn, grid):
    """grid 중 mean_cost가 가장 작은 threshold."""
    costs = [mean_cost(p, y, g, c_fp, c_fn) for g in grid]
    return float(np.asarray(grid)[int(np.argmin(costs))])
