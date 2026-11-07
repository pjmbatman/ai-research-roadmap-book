import numpy as np


def summarize(scores):
    """점수 배열의 {"n", "mean", "sd", "se"}. sd는 n-1로 나눈 표본 표준편차, se = sd / sqrt(n)."""
    raise NotImplementedError("평균, 표본 표준편차, 평균의 표준오차를 계산한다")


def paired_summary(a, b, t_crit):
    """같은 seed의 점수 쌍에서 d = b - a를 요약한 {"mean_diff", "se", "lo", "hi", "wins_b"}."""
    raise NotImplementedError("차이 d_i = b_i - a_i의 평균과 표준오차로 t 구간을 만든다")
