import numpy as np


def hypervolume_2d(F, ref):
    """두 목적(모두 최소화)의 점 집합이 지배하는 영역의 넓이(hypervolume).

    F: (m, 2) 목적값, ref: (2,) 기준점(모든 목적에서 나쁜 쪽 경계)
    반환: 점들이 지배하고 기준점보다 좋은 영역 [f1, ref1] x [f2, ref2]의 합집합의 넓이
    """
    F, ref = np.asarray(F, float), np.asarray(ref, float)
    F = F[np.all(F < ref, axis=1)]                          # 기준점 밖의 점은 기여하지 않는다
    F = F[np.argsort(F[:, 0])]                              # f1 오름차순
    area, prev_f2 = 0.0, ref[1]
    for f1, f2 in F:
        if f2 < prev_f2:                                    # 지배당한 점은 건너뛴다
            area += (ref[0] - f1) * (prev_f2 - f2)
            prev_f2 = f2
    return float(area)
