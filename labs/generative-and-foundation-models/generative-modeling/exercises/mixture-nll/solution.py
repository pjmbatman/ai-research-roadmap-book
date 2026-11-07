import numpy as np
from scipy.special import logsumexp


def mixture_nll(x, centers, sd, weights=None):
    """등방 가우시안 혼합 p(x) = sum_k w_k N(x; mu_k, sd^2 I)에서 x의 평균 음의 로그우도(nats)를 반환한다.

    x: (N, d), centers: (K, d), sd: 양수 스칼라, weights: (K,) 또는 None(균등)
    """
    x = np.asarray(x, dtype=float)
    centers = np.asarray(centers, dtype=float)
    K, d = centers.shape
    w = np.full(K, 1.0 / K) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)          # (N, K)
    log_comp = -d2 / (2 * sd**2) - 0.5 * d * np.log(2 * np.pi * sd**2)  # (N, K)
    log_p = logsumexp(np.log(w)[None, :] + log_comp, axis=1)            # (N,)
    return float(-log_p.mean())
