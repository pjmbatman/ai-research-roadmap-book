import math


def hoeffding_epsilon(n, delta, k=1):
    """2k exp(-2 n eps²) = delta를 만족하는 eps."""
    raise NotImplementedError("epsilon을 계산하세요")


def sample_size(epsilon, delta, k=1):
    """2k exp(-2 n eps²) <= delta를 만족하는 최소 정수 n."""
    raise NotImplementedError("필요한 표본 수를 계산하세요")
