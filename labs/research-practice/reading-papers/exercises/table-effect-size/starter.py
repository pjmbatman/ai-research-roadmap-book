import numpy as np
from scipy import stats


def sd_from_se(se, n):
    """표준오차 se와 실행 횟수 n에서 실행 간 표준편차를 구한다."""
    raise NotImplementedError("SD = SE * sqrt(n)")


def welch_ci(m1, s1, n1, m2, s2, n2, level=0.95):
    """차이 m2 - m1과 Welch 신뢰구간 (diff, lo, hi). s1, s2는 표준편차."""
    raise NotImplementedError("Welch 표준오차와 자유도로 t 구간을 만드세요")


def cohens_d(m1, s1, n1, m2, s2, n2):
    """합동 표준편차로 나눈 표준화 차이 (m2 - m1) / s_pooled."""
    raise NotImplementedError("합동 표준편차를 계산하세요")
