import numpy as np


def bh_reject(pvals, q=0.05):
    """Benjamini–Hochberg 절차. 입력과 같은 순서의 boolean 배열(기각 = True)을 반환한다."""
    raise NotImplementedError("p_(k) <= k q / m을 만족하는 가장 큰 k까지 기각한다")
