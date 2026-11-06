import numpy as np


def gaussian_nll(y, mu, sigma):
    """평균 가우시안 NLL. sigma <= 0이 있으면 ValueError."""
    raise NotImplementedError


def best_sigma(residuals, grid):
    """모든 샘플에 같은 sigma를 쓸 때 평균 NLL을 최소화하는 grid의 값."""
    raise NotImplementedError
