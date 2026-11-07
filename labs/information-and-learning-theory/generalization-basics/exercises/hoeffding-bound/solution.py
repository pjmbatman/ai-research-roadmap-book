import math


def hoeffding_epsilon(n, delta, k=1):
    """2k exp(-2 n eps²) = delta를 만족하는 eps."""
    return math.sqrt(math.log(2 * k / delta) / (2 * n))


def sample_size(epsilon, delta, k=1):
    """2k exp(-2 n eps²) <= delta를 만족하는 최소 정수 n."""
    return math.ceil(math.log(2 * k / delta) / (2 * epsilon**2))
