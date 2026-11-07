import numpy as np


def crps_ensemble(samples, y):
    """표본으로 주어진 예측분포의 CRPS를 관측마다 계산한다.

    samples: (S, N) 관측마다 예측 표본 S개, y: (N,) 관측값
    반환: (N,) 관측마다의 CRPS
    """
    x = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float)
    term1 = np.mean(np.abs(x - y[None, :]), axis=0)                         # (N,) E|X - y|
    term2 = np.mean(np.abs(x[:, None, :] - x[None, :, :]), axis=(0, 1))     # (N,) E|X - X'|
    return term1 - 0.5 * term2
