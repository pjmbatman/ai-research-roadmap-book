import numpy as np


def bisection(f, a, b, xtol=1e-12, max_iter=200):
    """부호가 바뀌는 구간 [a, b]에서 f의 근을 이분법으로 찾는다.

    반환: (x, n_iter) — |x - 근| <= xtol, n_iter = 구간을 절반으로 줄인 횟수
    f(a), f(b)의 부호가 같으면 ValueError
    """
    raise NotImplementedError("중점의 부호를 보고 부호가 바뀌는 절반을 남기세요")
