import math

import numpy as np


def poisson_nll(lam, counts, exposure_s):
    """y_i ~ Poisson(lam * t_i)일 때 전체 음의 로그우도 -Σ log p(y_i | lam)."""
    raise NotImplementedError("Poisson 확률질량함수의 로그를 합한다")


def poisson_mle(counts, exposure_s):
    """poisson_nll을 최소화하는 계수율 lam [1/s]의 닫힌 해."""
    raise NotImplementedError("로그우도를 lam으로 미분해 0으로 둔다")
