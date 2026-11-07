import numpy as np


def log_partition(eta):
    """A(eta) = log sum_j exp(eta_j). eta: (K,). 큰 값에서도 overflow 없이 계산한다."""
    eta = np.asarray(eta, dtype=float)
    m = eta.max()
    return float(m + np.log(np.sum(np.exp(eta - m))))


def mean_params(eta):
    """범주별 확률 pi_k = exp(eta_k - A(eta)). 반환 shape (K,)."""
    eta = np.asarray(eta, dtype=float)
    return np.exp(eta - log_partition(eta))


def log_prob(eta, k):
    """log p(y = k | eta) = eta_k - A(eta)."""
    eta = np.asarray(eta, dtype=float)
    return float(eta[k] - log_partition(eta))
