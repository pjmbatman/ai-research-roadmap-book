import numpy as np


def bias_variance(preds, f_true):
    """preds (R, M), f_true (M,)에서 평가 지점 평균 (bias2, variance)를 반환한다. 분산은 R로 나눈다."""
    P = np.asarray(preds, dtype=float)              # (R, M)
    mean_pred = P.mean(axis=0)                      # (M,)
    bias2 = np.mean((mean_pred - np.asarray(f_true)) ** 2)
    variance = np.mean(P.var(axis=0))               # ddof=0: R로 나눈다
    return float(bias2), float(variance)
