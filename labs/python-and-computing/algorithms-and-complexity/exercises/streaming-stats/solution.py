import numpy as np


def streaming_mean_var(chunks):
    """1차원 배열 조각들을 한 번만 순회하며 전체 평균과 모집단 분산(ddof=0)을 구한다.

    chunks: 1차원 NumPy 배열을 내보내는 iterable
    반환: (mean, var) — float 두 개. 전체가 비어 있으면 ValueError.
    """
    n, mean, m2 = 0, 0.0, 0.0
    for chunk in chunks:
        c = np.asarray(chunk, dtype=float).ravel()
        nb = c.size
        if nb == 0:
            continue
        mb = float(c.mean())
        m2b = float(((c - mb) ** 2).sum())        # 조각 안에서는 두 단계(two-pass)로
        total = n + nb
        delta = mb - mean
        mean += delta * nb / total
        m2 += m2b + delta**2 * n * nb / total
        n = total
    if n == 0:
        raise ValueError("데이터가 비어 있습니다")
    return mean, m2 / n
