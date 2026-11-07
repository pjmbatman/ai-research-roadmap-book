import numpy as np


def streaming_mean_var(chunks):
    """1차원 배열 조각들을 한 번만 순회하며 전체 평균과 모집단 분산(ddof=0)을 구한다.

    chunks: 1차원 NumPy 배열을 내보내는 iterable
    반환: (mean, var) — float 두 개. 전체가 비어 있으면 ValueError.
    """
    raise NotImplementedError("조각마다 (개수, 평균, 편차 제곱합)을 합치세요")
