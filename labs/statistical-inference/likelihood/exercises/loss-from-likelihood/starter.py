import numpy as np


def gaussian_nll(y, mu, sigma):
    """y ~ N(mu, sigma²)의 평균 음의 로그우도 (정규화 상수 포함)."""
    raise NotImplementedError("가우시안 밀도의 로그를 평균한다")


def laplace_nll(y, mu, b):
    """y ~ Laplace(mu, b)의 평균 음의 로그우도 (정규화 상수 포함)."""
    raise NotImplementedError("Laplace 밀도 exp(-|y-mu|/b) / (2b)의 로그를 평균한다")


def bernoulli_nll(y, p):
    """y ∈ {0, 1}, P[y=1] = p의 평균 음의 로그우도."""
    raise NotImplementedError("y=1이면 log p, y=0이면 log(1-p)")
