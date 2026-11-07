import numpy as np


def jacobian(f, x, eps=1e-6):
    """중앙차분으로 f의 Jacobian을 근사한다.

    f: (n,) -> (m,) 또는 스칼라
    x: (n,)
    반환: (m, n). f가 스칼라를 돌려주면 (1, n)
    """
    x = np.asarray(x, dtype=float)
    cols = []
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = eps
        diff = np.atleast_1d(f(x + e)) - np.atleast_1d(f(x - e))   # (m,)
        cols.append(diff / (2 * eps))
    return np.stack(cols, axis=1)                                   # (m, n)
