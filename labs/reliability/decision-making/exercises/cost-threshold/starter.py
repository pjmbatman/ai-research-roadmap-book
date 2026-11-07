import numpy as np


def bayes_threshold(c_fp, c_fn):
    """보정된 확률에서 기대 비용을 최소화하는 경보 threshold."""
    raise NotImplementedError("c_fp / (c_fp + c_fn)")


def mean_cost(p, y, thr, c_fp, c_fn):
    """p > thr이면 경보. y = 1 고장, 0 정상. 사례당 평균 비용."""
    raise NotImplementedError("오경보와 미탐지에 각각 비용을 매겨 평균 내세요")


def best_threshold(p, y, c_fp, c_fn, grid):
    """grid 중 mean_cost가 가장 작은 threshold."""
    raise NotImplementedError("grid의 각 값에서 mean_cost를 계산하세요")
