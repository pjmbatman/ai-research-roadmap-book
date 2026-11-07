import numpy as np


def chunked_mean_var(chunks):
    """덩어리를 한 번씩만 읽어 전체의 개수·평균·모분산을 구한다.

    chunks: 1차원 배열을 차례로 내놓는 iterable (한 번만 순회 가능)
    반환: (n, mean, var)  — n: int, mean: float, var: float (ddof=0)
    """
    raise NotImplementedError("덩어리마다 (n, 평균, M2)를 구해 병합 공식으로 합치세요")
