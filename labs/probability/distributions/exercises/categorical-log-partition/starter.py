import numpy as np


def log_partition(eta):
    """A(eta) = log sum_j exp(eta_j). eta: (K,). 큰 값에서도 overflow 없이 계산한다."""
    raise NotImplementedError("최댓값을 빼고 log-sum-exp를 계산한다")


def mean_params(eta):
    """범주별 확률 pi_k = exp(eta_k - A(eta)). 반환 shape (K,)."""
    raise NotImplementedError("log_partition을 써서 softmax를 계산한다")


def log_prob(eta, k):
    """log p(y = k | eta) = eta_k - A(eta)."""
    raise NotImplementedError("log_partition을 써서 계산한다")
