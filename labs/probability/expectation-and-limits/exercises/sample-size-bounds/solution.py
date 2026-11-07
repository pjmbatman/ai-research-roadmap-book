import math


def chebyshev_n(t, delta, sigma):
    """Chebyshev 경계 sigma^2 / (n t^2) <= delta를 만족하는 가장 작은 정수 n."""
    n = math.ceil(sigma**2 / (delta * t**2))
    while n > 1 and sigma**2 / ((n - 1) * t**2) <= delta:    # 부동소수점 올림 오차 보정
        n -= 1
    return n


def hoeffding_n(t, delta, a, b):
    """Hoeffding 경계 2 exp(-2 n t^2 / (b - a)^2) <= delta를 만족하는 가장 작은 정수 n."""
    n = math.ceil((b - a) ** 2 * math.log(2 / delta) / (2 * t**2))
    while n > 1 and 2 * math.exp(-2 * (n - 1) * t**2 / (b - a) ** 2) <= delta:
        n -= 1
    return n
