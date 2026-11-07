import numpy as np
from scipy.special import logsumexp


def e_step(x, pi, mu, sd):
    """책임도 resp (N, K)와 주변 로그우도 loglik(스칼라)를 반환한다.

    x: (N,), pi, mu, sd: (K,)
    """
    raise NotImplementedError("로그 공간에서 책임도와 로그우도를 계산하세요")


def m_step(x, resp):
    """책임도로 가중한 MLE (pi, mu, sd)를 반환한다. 각각 (K,)."""
    raise NotImplementedError("가중 비율·평균·표준편차를 계산하세요")
