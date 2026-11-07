import numpy as np


def gaussian_nll(beta, X, y, sigma):
    """y_i ~ N(x_i^T beta, sigma_i^2)의 평균 NLL [nat]. X: (N, p), y: (N,), sigma: (N,), beta: (p,)"""
    raise NotImplementedError("평균 음의 로그우도를 계산하세요")


def weighted_lstsq(X, y, sigma):
    """gaussian_nll을 최소화하는 beta (p,)를 반환한다."""
    raise NotImplementedError("가중 최소제곱 해를 계산하세요")
