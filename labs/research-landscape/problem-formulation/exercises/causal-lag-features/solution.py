import numpy as np


def lag_features(x, lags):
    """F[t, j] = x[t - lags[j]]인 lag 특징 행렬을 만든다. 값이 없는 칸은 NaN.

    x: (T,) 시계열
    lags: 0 이상의 정수 목록. 음수가 있으면 ValueError
    반환: (T, len(lags)) 배열
    """
    x = np.asarray(x, dtype=float)                 # (T,)
    T = len(x)
    if any(int(l) < 0 for l in lags):
        raise ValueError("음수 lag은 미래 값을 쓴다")
    F = np.full((T, len(lags)), np.nan)            # (T, L)
    for j, l in enumerate(lags):
        l = int(l)
        if l < T:
            F[l:, j] = x[: T - l]
    return F
