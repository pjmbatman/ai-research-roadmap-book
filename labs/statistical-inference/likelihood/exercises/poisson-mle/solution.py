import math

import numpy as np


def poisson_nll(lam, counts, exposure_s):
    """y_i ~ Poisson(lam * t_i)일 때 전체 음의 로그우도 -Σ log p(y_i | lam)."""
    y = np.asarray(counts, dtype=float)
    mu = lam * np.asarray(exposure_s, dtype=float)       # 측정마다 기대 계수
    log_fact = np.array([math.lgamma(v + 1) for v in y])
    return float(-np.sum(y * np.log(mu) - mu - log_fact))


def poisson_mle(counts, exposure_s):
    """poisson_nll을 최소화하는 계수율 lam [1/s]의 닫힌 해.

    d/dlam Σ (y_i log(lam t_i) - lam t_i) = Σ y_i / lam - Σ t_i = 0  →  lam = Σ y_i / Σ t_i
    """
    return float(np.sum(counts) / np.sum(exposure_s))
