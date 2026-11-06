import numpy as np


def pinball_loss(y, q, tau):
    """평균 pinball loss. u = y - q, rho_tau(u) = u * (tau - 1[u < 0])"""
    u = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return float(np.mean(u * (tau - (u < 0))))


def best_constant(y, tau, grid):
    """grid 중에서 평균 pinball loss를 최소화하는 값을 반환한다."""
    losses = [pinball_loss(y, q, tau) for q in grid]
    return float(np.asarray(grid)[int(np.argmin(losses))])
