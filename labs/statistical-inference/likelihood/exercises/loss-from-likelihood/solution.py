import numpy as np


def gaussian_nll(y, mu, sigma):
    """y ~ N(mu, sigma²)의 평균 음의 로그우도 (정규화 상수 포함)."""
    r = np.asarray(y, dtype=float) - mu
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + r**2 / (2 * sigma**2)))


def laplace_nll(y, mu, b):
    """y ~ Laplace(mu, b)의 평균 음의 로그우도 (정규화 상수 포함)."""
    r = np.asarray(y, dtype=float) - mu
    return float(np.mean(np.log(2 * b) + np.abs(r) / b))


def bernoulli_nll(y, p):
    """y ∈ {0, 1}, P[y=1] = p의 평균 음의 로그우도."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
