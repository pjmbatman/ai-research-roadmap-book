import numpy as np


def holm(pvals, alpha=0.05):
    """Holm 순차 기각 절차.

    pvals: (k,) p-value 배열
    반환: reject (k,) bool, p_adj (k,) float — 둘 다 입력과 같은 순서
    """
    raise NotImplementedError("정렬 → (k - j + 1)배 → 누적 최댓값 → 원래 순서로 되돌리기")
