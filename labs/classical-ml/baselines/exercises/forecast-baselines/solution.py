import numpy as np


def forecast_baselines(y, horizon, period):
    """학습 시계열 y (T,)로 다음 horizon개 시점을 예측하는 단순 baseline 네 개.

    반환: {"mean", "persistence", "seasonal", "drift"} → 각각 (horizon,) 배열
    """
    y = np.asarray(y, dtype=float)                       # (T,)
    T = len(y)
    h = np.arange(1, horizon + 1)                        # (H,)
    last_cycle = y[T - period:]                          # (p,)
    slope = (y[-1] - y[0]) / (T - 1)
    return {
        "mean": np.full(horizon, y.mean()),
        "persistence": np.full(horizon, y[-1]),
        "seasonal": last_cycle[(h - 1) % period],
        "drift": y[-1] + h * slope,
    }
