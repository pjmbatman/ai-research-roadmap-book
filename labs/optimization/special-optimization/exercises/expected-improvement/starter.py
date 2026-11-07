import numpy as np
from scipy.stats import norm


def expected_improvement(mu, sd, best, xi=0.0):
    """최소화 문제의 expected improvement.

    mu, sd: 대리모델의 예측 평균과 표준편차(스칼라 또는 같은 shape의 배열)
    best: 지금까지 관측한 최솟값, xi: 탐색을 위한 여유(0 이상)
    반환: E[max(best - xi - Y, 0)], Y ~ N(mu, sd^2). sd = 0이면 max(best - xi - mu, 0)
    """
    raise NotImplementedError("정규분포의 cdf와 pdf로 EI를 계산하세요")
