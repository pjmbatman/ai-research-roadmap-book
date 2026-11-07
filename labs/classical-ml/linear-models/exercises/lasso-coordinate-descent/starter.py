import numpy as np


def soft_threshold(z, t):
    """S(z, t) = sign(z) * max(|z| - t, 0), 원소별."""
    raise NotImplementedError("soft-thresholding을 구현하세요")


def lasso_cd(X, y, alpha, n_sweeps=500):
    """(1/2N)||y - X w||² + alpha ||w||_1을 좌표 하강법으로 최소화한다.

    X: (N, p), 열마다 평균 0. y: (N,). 반환: w (p,)
    """
    raise NotImplementedError("좌표 하강법을 구현하세요")
