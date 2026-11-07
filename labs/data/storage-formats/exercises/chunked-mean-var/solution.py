import numpy as np


def chunked_mean_var(chunks):
    """덩어리를 한 번씩만 읽어 전체의 개수·평균·모분산을 구한다.

    chunks: 1차원 배열을 차례로 내놓는 iterable (한 번만 순회 가능)
    반환: (n, mean, var)  — n: int, mean: float, var: float (ddof=0)
    """
    n, mean, m2 = 0, 0.0, 0.0
    for chunk in chunks:
        x = np.asarray(chunk, dtype=float).ravel()     # (n_b,)
        if x.size == 0:
            continue
        n_b, mean_b = x.size, float(x.mean())
        m2_b = float(((x - mean_b) ** 2).sum())        # 덩어리 안에서는 평균을 먼저 뺀다
        total = n + n_b
        delta = mean_b - mean
        mean = mean + delta * n_b / total
        m2 = m2 + m2_b + delta**2 * n * n_b / total
        n = total
    return n, mean, m2 / n
