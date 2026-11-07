import numpy as np


def rolling_origin_errors(y, forecaster, initial, horizon, step=1):
    """원점을 옮겨 가며 forecaster(y[:o+1])의 오차(예측 - 실제)를 모은다. 반환: (원점 수, horizon)"""
    y = np.asarray(y, dtype=float)
    rows = []
    for o in range(initial - 1, len(y) - horizon, step):      # o + horizon <= len(y) - 1
        pred = np.asarray(forecaster(y[: o + 1]), dtype=float)  # 과거만 넘긴다
        rows.append(pred - y[o + 1: o + 1 + horizon])
    return np.array(rows)


def horizon_rmse(errors):
    """horizon(열)별 RMSE. errors: (원점 수, horizon). 반환: (horizon,)"""
    return np.sqrt(np.mean(np.asarray(errors) ** 2, axis=0))
