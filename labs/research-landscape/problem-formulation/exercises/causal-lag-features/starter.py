import numpy as np


def lag_features(x, lags):
    """F[t, j] = x[t - lags[j]]인 lag 특징 행렬을 만든다. 값이 없는 칸은 NaN.

    x: (T,) 시계열
    lags: 0 이상의 정수 목록. 음수가 있으면 ValueError
    반환: (T, len(lags)) 배열
    """
    raise NotImplementedError("lag 특징 행렬을 만드세요")
