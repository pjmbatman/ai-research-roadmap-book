import numpy as np
from scipy.stats import norm


def ucb(mu, sd, beta=2.0):
    """최대화 문제의 UCB = mu + beta * sd.  mu, sd: (C,) -> (C,)"""
    return np.asarray(mu, dtype=float) + beta * np.asarray(sd, dtype=float)


def expected_improvement(mu, sd, best):
    """최대화 문제의 EI = E[max(Y - best, 0)], Y ~ N(mu, sd^2).

    mu, sd: (C,) 예측 평균·표준편차 (sd는 0일 수 있다)
    best: 지금까지의 최고 측정값
    반환: (C,) 0 이상인 EI
    """
    mu, sd = np.broadcast_arrays(np.asarray(mu, dtype=float), np.asarray(sd, dtype=float))
    gain = mu - best                                      # (C,)
    ei = np.maximum(gain, 0.0)                            # sd = 0이면 개선이 확정적
    pos = sd > 0
    z = gain[pos] / sd[pos]
    ei[pos] = gain[pos] * norm.cdf(z) + sd[pos] * norm.pdf(z)
    return np.maximum(ei, 0.0)                            # 반올림으로 생기는 아주 작은 음수 제거
