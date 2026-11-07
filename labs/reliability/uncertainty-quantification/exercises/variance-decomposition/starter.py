import numpy as np


def decompose(mu, var):
    """앙상블 예측분포의 분산을 aleatoric·epistemic 두 항으로 나눈다.

    mu: (M, N) 구성원별 평균, var: (M, N) 구성원별 분산
    반환: aleatoric (N,), epistemic (N,), total (N,)
    """
    raise NotImplementedError("전분산 법칙의 두 항을 계산하세요")
