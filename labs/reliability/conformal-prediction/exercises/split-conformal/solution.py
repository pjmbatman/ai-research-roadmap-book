import numpy as np


def conformal_quantile(scores, alpha):
    """보정 점수에서 ceil((n + 1)(1 - alpha))번째로 작은 값. 그 순위가 n보다 크면 np.inf."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[k - 1]) if k <= n else np.inf


def conformal_interval(pred_cal, y_cal, pred_test, alpha):
    """절댓값 잔차 점수로 만든 split conformal 구간 (lower, upper)."""
    scores = np.abs(np.asarray(y_cal, dtype=float) - np.asarray(pred_cal, dtype=float))
    q = conformal_quantile(scores, alpha)
    pred_test = np.asarray(pred_test, dtype=float)
    return pred_test - q, pred_test + q
