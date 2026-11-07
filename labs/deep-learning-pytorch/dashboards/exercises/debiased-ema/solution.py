import numpy as np


def smooth(values, weight):
    """보정한 지수이동평균. NaN·inf는 원본 그대로 두고 상태를 바꾸지 않는다.

    values: (T,) 기록된 값, weight: 0 <= w < 1
    반환: (T,) float 배열
    """
    x = np.asarray(values, dtype=float)
    out = x.copy()
    finite = x[np.isfinite(x)]
    if x.size == 0 or (finite.size == x.size and np.all(x == x[0])):
        return out
    s, k = 0.0, 0
    for i, v in enumerate(x):
        if not np.isfinite(v):
            continue                                  # 원본 값을 그대로 둔다
        s = weight * s + (1 - weight) * v
        k += 1
        out[i] = s / (1 - weight ** k) if weight != 1 else s
    return out
