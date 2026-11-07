import numpy as np


def trend_test(y):
    """최소제곱 추세와, 잔차의 lag-1 자기상관으로 보정한 표준오차.

    y: (n,)
    반환: slope, se_naive, n_eff, se_adj
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    t = np.arange(n) - (n - 1) / 2                       # (n,) 가운데로 옮긴 시점
    b = (t * y).sum() / (t**2).sum()
    e = y - y.mean() - b * t                             # 잔차
    se = np.sqrt((e**2).sum() / (n - 2) / (t**2).sum())
    r1 = (e[1:] * e[:-1]).sum() / (e**2).sum()
    n_eff = float(np.clip(n * (1 - r1) / (1 + r1), 3, n))
    return float(b), float(se), n_eff, float(se * np.sqrt((n - 2) / (n_eff - 2)))
