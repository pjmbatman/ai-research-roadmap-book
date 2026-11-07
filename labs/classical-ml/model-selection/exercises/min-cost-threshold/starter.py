import numpy as np


def best_threshold(y, p, c_fn, c_fp):
    """총비용 c_fn * FN + c_fp * FP를 최소화하는 threshold를 고른다.

    y: (N,) 0 또는 1, p: (N,) 양성 점수. p >= t이면 양성으로 판정한다.
    반환: (t, cost). 후보는 np.unique(p)와 np.inf, 비용이 같으면 가장 작은 t.
    """
    raise NotImplementedError("후보 threshold마다 총비용을 계산하세요")
