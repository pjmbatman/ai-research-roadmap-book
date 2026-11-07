import numpy as np
from scipy.special import logsumexp


def mixture_nll(x, centers, sd, weights=None):
    """등방 가우시안 혼합 p(x) = sum_k w_k N(x; mu_k, sd^2 I)에서 x의 평균 음의 로그우도(nats)를 반환한다.

    x: (N, d), centers: (K, d), sd: 양수 스칼라, weights: (K,) 또는 None(균등)
    """
    raise NotImplementedError("로그 공간에서 성분을 더해 평균 NLL을 계산하세요")
