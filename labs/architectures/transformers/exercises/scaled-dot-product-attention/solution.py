import numpy as np


def attention(Q, K, V, mask=None):
    """scaled dot-product attention.

    Q: (..., T, D), K: (..., S, D), V: (..., S, Dv)
    mask: None 또는 bool 배열 (T, S) / (..., T, S). True인 칸만 본다.
    반환: (output (..., T, Dv), weights (..., T, S))
    """
    Q, K, V = (np.asarray(a, dtype=float) for a in (Q, K, V))
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])     # (..., T, S)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)            # 수치 안정
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)                           # (..., T, S)
    return w @ V, w
