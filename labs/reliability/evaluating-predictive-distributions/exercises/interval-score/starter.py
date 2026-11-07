import numpy as np


def interval_score(lower, upper, y, alpha):
    """중심 (1 - alpha) 예측 구간 [lower, upper]의 평균 interval score."""
    raise NotImplementedError("폭 + 벗어난 거리 × 2/alpha 를 계산하세요")


def best_symmetric_halfwidth(y, center, alpha, grid):
    """구간 [center - h, center + h]의 평균 interval score를 가장 작게 만드는 h (grid 중에서)."""
    raise NotImplementedError("grid의 각 h에 대해 interval_score를 계산해 가장 작은 것을 고르세요")
