import numpy as np


def latin_hypercube(n, d, seed):
    """[0, 1)^d에서 Latin hypercube 표본 n개를 뽑는다.

    n: 표본 수(각 축의 구간 수), d: 차원, seed: 난수 시드
    반환: (n, d) 배열. 각 열은 n개 구간에 한 점씩
    """
    raise NotImplementedError("축마다 순열을 만들고 구간 안에서 무작위 위치를 더하세요")
