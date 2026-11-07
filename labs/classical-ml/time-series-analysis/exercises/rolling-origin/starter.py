import numpy as np


def rolling_origin_errors(y, forecaster, initial, horizon, step=1):
    """원점을 옮겨 가며 forecaster(y[:o+1])의 오차(예측 - 실제)를 모은다. 반환: (원점 수, horizon)"""
    raise NotImplementedError("원점마다 과거만 넘겨 예측하고 오차를 모으세요")


def horizon_rmse(errors):
    """horizon(열)별 RMSE. errors: (원점 수, horizon). 반환: (horizon,)"""
    raise NotImplementedError("열마다 RMSE를 계산하세요")
