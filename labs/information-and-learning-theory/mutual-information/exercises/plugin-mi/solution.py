import numpy as np


def plugin_mi(counts):
    """결합 도수표 counts (A, B)에서 plug-in 상호정보량 [nat]을 계산한다. 도수 0인 칸은 0으로 둔다."""
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()                            # (A, B)
    px = p.sum(axis=1, keepdims=True)          # (A, 1)
    py = p.sum(axis=0, keepdims=True)          # (1, B)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / (px @ py)[m])))
