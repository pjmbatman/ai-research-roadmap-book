import numpy as np


def elbo_mc(log_joint, m, s, eps):
    """theta = m + s * eps로 추정한 ELBO = mean(log_joint(theta) - log q(theta)). s <= 0이면 ValueError."""
    if s <= 0:
        raise ValueError("s는 양수여야 합니다")
    eps = np.asarray(eps, dtype=float)
    theta = m + s * eps                               # (K,) 재매개변수화
    log_q = -0.5 * eps**2 - np.log(s * np.sqrt(2 * np.pi))
    return float(np.mean(log_joint(theta) - log_q))


def kl_gaussians(m1, s1, m2, s2):
    """KL(N(m1, s1^2) || N(m2, s2^2))"""
    return float(np.log(s2 / s1) + (s1**2 + (m1 - m2) ** 2) / (2 * s2**2) - 0.5)
