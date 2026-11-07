import numpy as np
from scipy.special import logsumexp


def e_step(x, pi, mu, sd):
    """책임도 resp (N, K)와 주변 로그우도 loglik(스칼라)를 반환한다.

    x: (N,), pi, mu, sd: (K,)
    """
    x = np.asarray(x, dtype=float)[:, None]                                   # (N, 1)
    pi, mu, sd = (np.asarray(v, dtype=float) for v in (pi, mu, sd))
    log_joint = np.log(pi) - 0.5 * np.log(2 * np.pi * sd**2) - (x - mu) ** 2 / (2 * sd**2)   # (N, K)
    log_px = logsumexp(log_joint, axis=1, keepdims=True)                       # (N, 1)
    return np.exp(log_joint - log_px), float(log_px.sum())


def m_step(x, resp):
    """책임도로 가중한 MLE (pi, mu, sd)를 반환한다. 각각 (K,)."""
    x = np.asarray(x, dtype=float)
    resp = np.asarray(resp, dtype=float)
    Nk = resp.sum(axis=0)                                                      # (K,)
    pi = Nk / len(x)
    mu = resp.T @ x / Nk
    sd = np.sqrt(np.sum(resp * (x[:, None] - mu) ** 2, axis=0) / Nk)
    return pi, mu, sd
