import numpy as np


def variance_decomposition(y):
    """배치별 측정값 y (G, n)의 분산을 배치 안·배치 간으로 나눈다.

    반환: (total, within, between) — 세 개의 float. 모두 ddof=0 분산.
    """
    raise NotImplementedError("total, within, between을 계산하세요")
