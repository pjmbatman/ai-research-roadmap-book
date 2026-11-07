import numpy as np


def equivariant_layer(H, W_self, W_pool, b):
    """순열 등변 층: H W_self + mean_n(H) W_pool + b.

    H: (B, N, D_in), W_self: (D_in, D_out), W_pool: (D_in, D_out), b: (D_out,)
    반환: (B, N, D_out)
    """
    H = np.asarray(H, dtype=float)
    pooled = H.mean(axis=1, keepdims=True)              # (B, 1, D_in), 원소 축 평균
    return H @ W_self + pooled @ W_pool + b             # (B, N, D_out) + (B, 1, D_out) + (D_out,)
