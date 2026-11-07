import numpy as np


def bisection(f, a, b, xtol=1e-12, max_iter=200):
    """부호가 바뀌는 구간 [a, b]에서 f의 근을 이분법으로 찾는다.

    반환: (x, n_iter) — |x - 근| <= xtol, n_iter = 구간을 절반으로 줄인 횟수
    f(a), f(b)의 부호가 같으면 ValueError
    """
    sa, sb = np.sign(f(a)), np.sign(f(b))
    if sa == 0:
        return a, 0
    if sb == 0:
        return b, 0
    if sa == sb:
        raise ValueError("f(a)와 f(b)의 부호가 같습니다: 근을 감싸는 구간이 아닙니다")
    for k in range(max_iter):
        if b - a <= 2 * xtol:
            return 0.5 * (a + b), k
        m = 0.5 * (a + b)
        sm = np.sign(f(m))
        if sm == 0:
            return m, k + 1
        if sm == sa:
            a, sa = m, sm          # 근은 [m, b]에
        else:
            b = m                  # 근은 [a, m]에
    return 0.5 * (a + b), max_iter
