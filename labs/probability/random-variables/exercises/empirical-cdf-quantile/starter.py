import numpy as np


def ecdf(samples, x):
    """경험적 CDF F_n(x) = (x 이하인 표본의 수) / n. x와 같은 shape으로 반환한다."""
    raise NotImplementedError("정렬한 표본에서 x 이하인 원소의 개수를 센다")


def quantile(samples, p):
    """일반화 역함수 Q_n(p) = min{x_i : F_n(x_i) >= p}, 0 < p <= 1. p와 같은 shape으로 반환한다."""
    raise NotImplementedError("정렬한 표본의 ceil(n p)번째 값(1부터 셀 때)을 고른다")
