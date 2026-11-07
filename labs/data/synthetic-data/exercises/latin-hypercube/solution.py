import numpy as np


def latin_hypercube(n, d, seed):
    """[0, 1)^d에서 Latin hypercube 표본 n개를 뽑는다.

    n: 표본 수(각 축의 구간 수), d: 차원, seed: 난수 시드
    반환: (n, d) 배열. 각 열은 n개 구간에 한 점씩
    """
    rng = np.random.default_rng(seed)
    strata = np.column_stack([rng.permutation(n) for _ in range(d)])   # (n, d) 축마다 다른 순열
    jitter = rng.random((n, d))                                         # (n, d) 구간 안의 위치
    return (strata + jitter) / n
