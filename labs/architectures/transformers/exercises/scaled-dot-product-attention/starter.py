import numpy as np


def attention(Q, K, V, mask=None):
    """scaled dot-product attention.

    Q: (..., T, D), K: (..., S, D), V: (..., S, Dv)
    mask: None 또는 bool 배열 (T, S) / (..., T, S). True인 칸만 본다.
    반환: (output (..., T, Dv), weights (..., T, S))
    """
    raise NotImplementedError("점수 → mask → softmax → 가중합 순서로 계산하세요")
