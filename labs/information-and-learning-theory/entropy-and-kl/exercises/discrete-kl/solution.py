import numpy as np


def cross_entropy(p, q):
    """H(p, q) = -Σ p log q [nat]. p_k = 0인 항은 0, p_k > 0이고 q_k = 0이면 inf.

    p, q: (K,) 확률 벡터
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = p > 0                              # 0 log q = 0
    if np.any(q[m] == 0):
        return np.inf
    return float(-np.sum(p[m] * np.log(q[m])))


def kl_divergence(p, q):
    """KL(p || q) = Σ p log(p / q) [nat]. p_k = 0인 항은 0, p_k > 0이고 q_k = 0이면 inf.

    p, q: (K,) 확률 벡터
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = p > 0
    if np.any(q[m] == 0):
        return np.inf
    return float(np.sum(p[m] * np.log(p[m] / q[m])))
