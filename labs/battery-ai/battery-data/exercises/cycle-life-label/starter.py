import numpy as np


def cycle_life(cycle, capacity_ah, ref_ah, frac=0.8, window=5):
    """후행 이동 중앙값으로 평활한 용량이 frac * ref_ah 아래로 처음 떨어지는 사이클을 찾는다.

    cycle: (N,) 사이클 번호, capacity_ah: (N,) 용량 [Ah] (측정 실패는 NaN)
    반환: (life, observed). 끝까지 떨어지지 않으면 (cycle[-1], False)
    """
    raise NotImplementedError("NaN을 뺀 후행 중앙값으로 첫 교차를 찾으세요")
