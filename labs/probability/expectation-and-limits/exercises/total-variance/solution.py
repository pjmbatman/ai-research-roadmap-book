import numpy as np


def variance_decomposition(y):
    """배치별 측정값 y (G, n)의 분산을 배치 안·배치 간으로 나눈다.

    반환: (total, within, between) — 세 개의 float. 모두 ddof=0 분산.
    """
    y = np.asarray(y, dtype=float)       # (G, n)
    total = y.var()                      # 모두 모은 분산
    within = y.var(axis=1).mean()        # E[Var(Y | 배치)]
    between = y.mean(axis=1).var()       # Var(E[Y | 배치])
    return float(total), float(within), float(between)
