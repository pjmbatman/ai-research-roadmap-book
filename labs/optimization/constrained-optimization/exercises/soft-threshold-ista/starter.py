import numpy as np


def soft_threshold(v, t):
    """t * ||.||_1의 근위 연산: 성분마다 sign(v) * max(|v| - t, 0)."""
    raise NotImplementedError("soft thresholding을 구현하세요")


def ista(A, y, lam, steps=2000):
    """0.5 ||A w - y||^2 + lam ||w||_1 을 proximal gradient로 최소화한다.

    A: (N, D), y: (N,), lam: 0 이상
    반환: w (D,)
    """
    raise NotImplementedError("gradient step 뒤에 soft thresholding을 적용하는 반복을 구현하세요")
