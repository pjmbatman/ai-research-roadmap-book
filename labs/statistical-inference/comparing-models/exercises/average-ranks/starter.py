import numpy as np


def average_ranks(scores, lower_is_better=True):
    """(N, k) 점수표에서 데이터셋마다 순위를 매기고(동점은 평균 순위) 방법별 평균 순위 (k,)를 반환한다."""
    raise NotImplementedError("행마다 순위를 매기고, 동점에는 평균 순위를 준다")


def friedman_chi2(avg_ranks, n_datasets):
    """평균 순위로 Friedman 통계량을 계산한다."""
    raise NotImplementedError("12N/(k(k+1)) * (Σ R_j² - k(k+1)²/4)")


def critical_difference(k, n_datasets, q_alpha):
    """Nemenyi 검정의 critical difference."""
    raise NotImplementedError("q_alpha * sqrt(k(k+1) / (6N))")
