import numpy as np
from scipy.linalg import sqrtm


def frechet_from_stats(mu1, S1, mu2, S2):
    """두 가우시안 N(mu1, S1), N(mu2, S2) 사이의 Fréchet 거리 d².

    mu1, mu2: (d,), S1, S2: (d, d)
    반환: float
    """
    mu1, mu2 = np.atleast_1d(np.asarray(mu1, float)), np.atleast_1d(np.asarray(mu2, float))
    S1, S2 = np.atleast_2d(np.asarray(S1, float)), np.atleast_2d(np.asarray(S2, float))
    covmean = sqrtm(S1 @ S2).real                       # (S1 S2)^{1/2}
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(S1 + S2 - 2 * covmean))


def frechet_from_features(a, b):
    """특징 행렬 a (N, d), b (M, d)에 가우시안을 맞춘 뒤의 Fréchet 거리."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return frechet_from_stats(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False))
