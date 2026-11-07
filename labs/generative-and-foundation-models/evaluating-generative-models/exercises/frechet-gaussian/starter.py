import numpy as np
from scipy.linalg import sqrtm


def frechet_from_stats(mu1, S1, mu2, S2):
    """두 가우시안 N(mu1, S1), N(mu2, S2) 사이의 Fréchet 거리 d².

    mu1, mu2: (d,), S1, S2: (d, d)
    반환: float
    """
    raise NotImplementedError("Fréchet 거리를 계산하세요")


def frechet_from_features(a, b):
    """특징 행렬 a (N, d), b (M, d)에 가우시안을 맞춘 뒤의 Fréchet 거리."""
    raise NotImplementedError("표본 평균·공분산을 구해 frechet_from_stats를 호출하세요")
