import numpy as np


def hypervolume_2d(F, ref):
    """두 목적(모두 최소화)의 점 집합이 지배하는 영역의 넓이(hypervolume).

    F: (m, 2) 목적값, ref: (2,) 기준점(모든 목적에서 나쁜 쪽 경계)
    반환: 점들이 지배하고 기준점보다 좋은 영역 [f1, ref1] x [f2, ref2]의 합집합의 넓이
    """
    raise NotImplementedError("지배 영역의 넓이를 계산하세요")
