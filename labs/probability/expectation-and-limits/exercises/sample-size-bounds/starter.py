import math


def chebyshev_n(t, delta, sigma):
    """Chebyshev 경계 sigma^2 / (n t^2) <= delta를 만족하는 가장 작은 정수 n."""
    raise NotImplementedError("Chebyshev 경계를 n에 대해 푸세요")


def hoeffding_n(t, delta, a, b):
    """Hoeffding 경계 2 exp(-2 n t^2 / (b - a)^2) <= delta를 만족하는 가장 작은 정수 n."""
    raise NotImplementedError("Hoeffding 경계를 n에 대해 푸세요")
