import numpy as np


def gd_quadratic(H, b, x0, lr, steps):
    """f(x) = x^T H x / 2 - b^T x 에 경사하강을 steps번 적용한 x를 반환한다.

    H: (n, n) 대칭 양의 정부호, b: (n,), x0: (n,)
    """
    raise NotImplementedError("x <- x - lr * (H x - b)를 반복하세요")


def max_stable_lr(H):
    """GD가 수렴하는 학습률의 상한(스칼라)."""
    raise NotImplementedError("H의 고윳값으로 상한을 구하세요")
