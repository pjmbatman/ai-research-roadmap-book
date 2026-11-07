import numpy as np


def factorial_effects(levels, y):
    """2^k 요인 설계의 주효과와 2차 교호작용.

    levels: (N, k) 배열, 원소는 0 또는 1. y: (N,) 결과.
    반환: {(j,): 주효과, (i, j): 교호작용 (i < j)}
    """
    raise NotImplementedError("부호 s = 2 * levels - 1로 대비(contrast)를 계산하세요")
