import numpy as np
from scipy import stats


def posterior(a0, b0, counts, exposure_s):
    """사전 Gamma(a0, b0)(shape, rate)와 계수 counts, 노출 시간 exposure_s [s]로 사후 Gamma(a, b)를 구한다.

    counts: (n,), exposure_s: (n,)
    반환: (a, b)
    """
    raise NotImplementedError("사후분포의 shape와 rate를 계산하세요")


def credible_interval(a, b, level=0.95):
    """Gamma(a, b)(shape, rate)의 등꼬리 신용구간 (lo, hi)를 반환한다."""
    raise NotImplementedError("분위수로 구간을 계산하세요")


def prob_rate_above(a, b, threshold):
    """Gamma(a, b)(shape, rate) 아래에서 Pr[λ > threshold]를 반환한다."""
    raise NotImplementedError("꼬리 확률을 계산하세요")
