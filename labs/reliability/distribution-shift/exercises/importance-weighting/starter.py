import numpy as np


def weighted_risk(losses, w):
    """자기 정규화 중요도 가중 평균 손실 sum(w * loss) / sum(w)."""
    raise NotImplementedError("가중 평균을 계산하세요")


def effective_sample_size(w):
    """유효 표본 수 (sum w)^2 / sum(w^2)."""
    raise NotImplementedError("유효 표본 수를 계산하세요")


def weighted_line_fit(x, y, w):
    """sum_i w_i (y_i - a - b x_i)^2 를 최소화하는 (a, b)."""
    raise NotImplementedError("가중 정규방정식을 푸세요")
