import numpy as np


def elbo_mc(log_joint, m, s, eps):
    """theta = m + s * eps로 추정한 ELBO = mean(log_joint(theta) - log q(theta)). s <= 0이면 ValueError."""
    raise NotImplementedError("재매개변수화한 표본에서 log_joint - log q의 평균을 구하세요")


def kl_gaussians(m1, s1, m2, s2):
    """KL(N(m1, s1^2) || N(m2, s2^2))"""
    raise NotImplementedError
