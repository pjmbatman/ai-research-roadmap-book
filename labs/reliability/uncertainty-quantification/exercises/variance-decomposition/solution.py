import numpy as np


def decompose(mu, var):
    """앙상블 예측분포의 분산을 aleatoric·epistemic 두 항으로 나눈다.

    mu: (M, N) 구성원별 평균, var: (M, N) 구성원별 분산
    반환: aleatoric (N,), epistemic (N,), total (N,)
    """
    mu, var = np.asarray(mu, dtype=float), np.asarray(var, dtype=float)
    aleatoric = var.mean(axis=0)                          # (N,) 구성원 분산의 평균
    epistemic = ((mu - mu.mean(axis=0)) ** 2).mean(axis=0)  # (N,) 구성원 평균의 분산 (ddof=0)
    return aleatoric, epistemic, aleatoric + epistemic
