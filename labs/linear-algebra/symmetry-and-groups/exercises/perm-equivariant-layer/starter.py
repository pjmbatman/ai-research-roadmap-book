import numpy as np


def equivariant_layer(H, W_self, W_pool, b):
    """순열 등변 층: H W_self + mean_n(H) W_pool + b.

    H: (B, N, D_in), W_self: (D_in, D_out), W_pool: (D_in, D_out), b: (D_out,)
    반환: (B, N, D_out)
    """
    raise NotImplementedError("원소 축 평균을 이용해 등변 층을 구현하세요")
