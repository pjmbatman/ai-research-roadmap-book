import numpy as np


def condition(mu, Sigma, obs_idx, x_obs):
    """x ~ N(mu, Sigma)에서 obs_idx 성분이 x_obs일 때 나머지 성분(인덱스 오름차순)의 조건부 평균·공분산.

    반환: (cond_mean (D-k,), cond_cov (D-k, D-k))
    """
    raise NotImplementedError("블록을 나누고 Schur complement 공식을 solve로 계산한다")
