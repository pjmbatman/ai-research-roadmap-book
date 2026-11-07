import numpy as np
from scipy.stats import norm


def ucb(mu, sd, beta=2.0):
    """최대화 문제의 UCB = mu + beta * sd.  mu, sd: (C,) -> (C,)"""
    raise NotImplementedError("UCB를 계산하세요")


def expected_improvement(mu, sd, best):
    """최대화 문제의 EI = E[max(Y - best, 0)], Y ~ N(mu, sd^2).

    mu, sd: (C,) 예측 평균·표준편차 (sd는 0일 수 있다)
    best: 지금까지의 최고 측정값
    반환: (C,) 0 이상인 EI
    """
    raise NotImplementedError("z = (mu - best) / sd로 EI를 계산하고 sd = 0인 경우를 따로 처리하세요")
