import numpy as np


def pareto_mask(F):
    """모든 목적을 최대화할 때 Pareto front에 있는 행을 표시한다.

    F: (N, k) 측정값. 행 = 실험, 열 = 목적(클수록 좋다)
    반환: (N,) 불리언. True = 어떤 실험에도 지배되지 않음
    """
    raise NotImplementedError("다른 실험에 지배되는 행을 찾아 False로 표시하세요")
