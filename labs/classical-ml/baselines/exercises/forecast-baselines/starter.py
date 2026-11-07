import numpy as np


def forecast_baselines(y, horizon, period):
    """학습 시계열 y (T,)로 다음 horizon개 시점을 예측하는 단순 baseline 네 개.

    반환: {"mean", "persistence", "seasonal", "drift"} → 각각 (horizon,) 배열
    """
    raise NotImplementedError("네 가지 baseline 예측을 계산하세요")
