import numpy as np


def numerical_grad(f, w, h=1e-5):
    """중심 차분으로 f의 gradient를 근사한다.

    f: (D,) 배열 → 스칼라
    w: (D,) 배열 (바꾸지 않는다)
    반환: (D,) 배열, i번째 성분은 (f(w + h e_i) - f(w - h e_i)) / (2h)
    """
    w = np.asarray(w, dtype=float)
    g = np.zeros_like(w)                 # (D,)
    for i in range(w.size):
        e = np.zeros_like(w)             # (D,)
        e[i] = h
        g[i] = (f(w + e) - f(w - e)) / (2 * h)
    return g
