import numpy as np
from scipy import stats


def sd_from_se(se, n):
    """표준오차 se와 실행 횟수 n에서 실행 간 표준편차를 구한다."""
    return float(se * np.sqrt(n))


def welch_ci(m1, s1, n1, m2, s2, n2, level=0.95):
    """차이 m2 - m1과 Welch 신뢰구간 (diff, lo, hi). s1, s2는 표준편차."""
    a, b = s1**2 / n1, s2**2 / n2
    se = np.sqrt(a + b)
    nu = (a + b) ** 2 / (a**2 / (n1 - 1) + b**2 / (n2 - 1))
    t = stats.t.ppf((1 + level) / 2, nu)
    diff = m2 - m1
    return float(diff), float(diff - t * se), float(diff + t * se)


def cohens_d(m1, s1, n1, m2, s2, n2):
    """합동 표준편차로 나눈 표준화 차이 (m2 - m1) / s_pooled."""
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return float((m2 - m1) / sp)
