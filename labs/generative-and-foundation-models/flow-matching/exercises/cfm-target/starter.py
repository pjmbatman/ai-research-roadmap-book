import numpy as np


def cfm_pair(x0, x1, t):
    """직선 조건부 경로의 표본 x_t와 목표 속도 u = x1 - x0.

    x0, x1: (N, d), t: (N,)
    반환: xt (N, d), u (N, d)
    """
    raise NotImplementedError("x_t와 u를 계산하세요")


def velocity_from_eps(xt, eps_hat, t):
    """직선 경로에서 ε-예측 E[x0 | x_t]를 속도 E[x1 - x0 | x_t]로 바꾼다.

    xt, eps_hat: (N, d), t: (N,), 0 < t <= 1
    반환: v (N, d)
    """
    raise NotImplementedError("속도를 계산하세요")
