import numpy as np


def posterior(x0, xt, t, betas):
    """q(x_{t-1} | x_t, x_0)의 (평균, 분산)을 반환한다. t는 1부터 센다(betas[t - 1] = beta_t)."""
    raise NotImplementedError("두 가우시안 인자를 곱해 사후 평균과 분산을 구하세요")
