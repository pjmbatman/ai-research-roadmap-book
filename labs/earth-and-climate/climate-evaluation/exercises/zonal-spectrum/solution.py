import numpy as np


def energy_ratio(fc, truth):
    """동서 방향 파수별 에너지 비 P_fc(k) / P_truth(k), k = 1..nlon/2, 와 비율이 처음 0.5 아래로 내려가는 k.

    fc, truth: (..., nlon)
    반환: ratio (nlon // 2,), k_half (int 또는 None)
    """
    fc, truth = np.asarray(fc, dtype=float), np.asarray(truth, dtype=float)
    nlon = fc.shape[-1]

    def power(x):                                         # (nlon // 2 + 1,)
        return (np.abs(np.fft.rfft(x.reshape(-1, nlon), axis=-1)) ** 2).mean(0) / nlon**2

    ratio = power(fc)[1:nlon // 2 + 1] / power(truth)[1:nlon // 2 + 1]
    below = np.flatnonzero(ratio < 0.5)
    return ratio, (int(below[0]) + 1 if below.size else None)
