import numpy as np


def leapfrog(theta, r, grad_log_p, eps, L):
    """leapfrog L걸음을 진행한 (theta, r). 입력 배열은 바꾸지 않는다."""
    theta = np.array(theta, dtype=float)              # 복사본
    r = np.array(r, dtype=float) + 0.5 * eps * grad_log_p(theta)
    for i in range(L):
        theta = theta + eps * r
        r = r + (eps if i < L - 1 else 0.5 * eps) * grad_log_p(theta)
    return theta, r


def hamiltonian(theta, r, log_p):
    """H = -log p(theta) + r·r / 2"""
    r = np.asarray(r, dtype=float)
    return float(-log_p(np.asarray(theta, dtype=float)) + 0.5 * r @ r)
