import numpy as np


def bh_reject(pvals, q=0.05):
    """Benjamini–Hochberg 절차. 입력과 같은 순서의 boolean 배열(기각 = True)을 반환한다."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m          # 정렬된 순위마다 조건, (m,)
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        k = np.nonzero(passed)[0].max() + 1                    # 조건을 만족하는 가장 큰 순위
        reject[order[:k]] = True
    return reject
