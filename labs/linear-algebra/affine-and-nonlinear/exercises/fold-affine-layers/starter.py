import numpy as np


def fold(W1, b1, W2, b2):
    """두 아핀 층 W2 (W1 x + b1) + b2를 하나의 (W, b)로 접는다.

    W1: (h, d), b1: (h,), W2: (m, h), b2: (m,)
    반환: W (m, d), b (m,)
    """
    raise NotImplementedError("W와 b를 계산하세요")
