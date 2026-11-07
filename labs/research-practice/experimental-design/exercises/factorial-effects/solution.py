import itertools

import numpy as np


def factorial_effects(levels, y):
    """2^k 요인 설계의 주효과와 2차 교호작용.

    levels: (N, k) 배열, 원소는 0 또는 1. y: (N,) 결과.
    반환: {(j,): 주효과, (i, j): 교호작용 (i < j)}
    """
    S = 2 * np.asarray(levels, dtype=float) - 1           # (N, k)
    y = np.asarray(y, dtype=float)
    k = S.shape[1]
    contrast = lambda s: y[s > 0].mean() - y[s < 0].mean()
    out = {(j,): float(contrast(S[:, j])) for j in range(k)}
    for i, j in itertools.combinations(range(k), 2):
        out[(i, j)] = float(contrast(S[:, i] * S[:, j]))
    return out
