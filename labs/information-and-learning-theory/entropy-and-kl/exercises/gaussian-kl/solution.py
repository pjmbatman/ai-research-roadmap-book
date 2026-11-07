import numpy as np


def gaussian_kl(mu1, sigma1, mu2, sigma2):
    """KL( N(mu1, sigma1²) || N(mu2, sigma2²) ) [nat]. 인자는 스칼라 또는 같은 shape의 배열."""
    mu1, sigma1, mu2, sigma2 = map(np.asarray, (mu1, sigma1, mu2, sigma2))
    return np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2) - 0.5
