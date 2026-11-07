import numpy as np


def conformal_quantile(scores, alpha):
    """보정 점수에서 ceil((n + 1)(1 - alpha))번째로 작은 값. 그 순위가 n보다 크면 np.inf."""
    raise NotImplementedError("순위 k를 계산하고 정렬한 점수에서 k번째 값을 고르세요")


def conformal_interval(pred_cal, y_cal, pred_test, alpha):
    """절댓값 잔차 점수로 만든 split conformal 구간 (lower, upper)."""
    raise NotImplementedError("보정 점수 |y - pred|로 q_hat을 구하고 pred_test ± q_hat을 반환하세요")
