import numpy as np


def holm(pvals, alpha=0.05):
    """Holm 순차 기각 절차.

    pvals: (k,) p-value 배열
    반환: reject (k,) bool, p_adj (k,) float — 둘 다 입력과 같은 순서
    """
    p = np.asarray(pvals, dtype=float)                       # (k,)
    k = len(p)
    order = np.argsort(p, kind="stable")                     # 작은 p부터
    scaled = (k - np.arange(k)) * p[order]                   # (k - j + 1) p_(j), j = 1..k
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    p_adj = np.empty(k)
    p_adj[order] = adj_sorted                                # 원래 순서로 되돌린다
    return p_adj <= alpha, p_adj
