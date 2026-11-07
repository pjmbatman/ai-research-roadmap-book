import numpy as np


def average_ranks(scores, lower_is_better=True):
    """(N, k) 점수표에서 데이터셋마다 순위를 매기고(동점은 평균 순위) 방법별 평균 순위 (k,)를 반환한다."""
    s = np.asarray(scores, dtype=float)
    if not lower_is_better:
        s = -s
    ranks = np.empty_like(s)
    for i, row in enumerate(s):
        for j, v in enumerate(row):
            below = np.sum(row < v)                 # 더 좋은 방법의 수
            ties = np.sum(row == v)                 # 자신을 포함한 동점 수
            ranks[i, j] = below + (ties + 1) / 2    # 순위 below+1, ..., below+ties의 평균
    return ranks.mean(axis=0)


def friedman_chi2(avg_ranks, n_datasets):
    """평균 순위로 Friedman 통계량을 계산한다."""
    r = np.asarray(avg_ranks, dtype=float)
    k = len(r)
    return float(12 * n_datasets / (k * (k + 1)) * (np.sum(r**2) - k * (k + 1) ** 2 / 4))


def critical_difference(k, n_datasets, q_alpha):
    """Nemenyi 검정의 critical difference."""
    return float(q_alpha * np.sqrt(k * (k + 1) / (6 * n_datasets)))
