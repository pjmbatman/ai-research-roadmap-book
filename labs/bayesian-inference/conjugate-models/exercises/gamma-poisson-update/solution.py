import numpy as np
from scipy import stats


def posterior(a0, b0, counts, exposure_s):
    """사전 Gamma(a0, b0)(shape, rate)와 계수 counts, 노출 시간 exposure_s [s]로 사후 Gamma(a, b)를 구한다.

    counts: (n,), exposure_s: (n,)
    반환: (a, b)
    """
    counts = np.asarray(counts, dtype=float)
    exposure_s = np.asarray(exposure_s, dtype=float)
    return a0 + counts.sum(), b0 + exposure_s.sum()


def credible_interval(a, b, level=0.95):
    """Gamma(a, b)(shape, rate)의 등꼬리 신용구간 (lo, hi)를 반환한다."""
    tail = (1 - level) / 2
    lo, hi = stats.gamma.ppf([tail, 1 - tail], a, scale=1 / b)
    return lo, hi


def prob_rate_above(a, b, threshold):
    """Gamma(a, b)(shape, rate) 아래에서 Pr[λ > threshold]를 반환한다."""
    return stats.gamma.sf(threshold, a, scale=1 / b)
