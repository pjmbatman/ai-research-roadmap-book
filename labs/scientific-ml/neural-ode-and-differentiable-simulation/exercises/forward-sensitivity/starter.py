import numpy as np


def euler_with_sensitivity(r, K, y0, h, steps):
    """y' = r y (1 - y/K)를 explicit Euler로 적분하며 s = dy/dr도 함께 계산한다.

    반환: y (steps+1,), s (steps+1,). y[0] = y0, s[0] = 0.
    """
    raise NotImplementedError("Euler 갱신과 민감도 갱신을 함께 구현하세요")
