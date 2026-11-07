import numpy as np


def cfm_pair(x0, x1, t):
    """직선 조건부 경로의 표본 x_t와 목표 속도 u = x1 - x0.

    x0, x1: (N, d), t: (N,)
    반환: xt (N, d), u (N, d)
    """
    x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
    t = np.asarray(t, float)[:, None]          # (N, 1): 표본마다 자기 시간
    xt = (1 - t) * x0 + t * x1                 # (N, d)
    u = x1 - x0                                # (N, d)
    return xt, u


def velocity_from_eps(xt, eps_hat, t):
    """직선 경로에서 ε-예측 E[x0 | x_t]를 속도 E[x1 - x0 | x_t]로 바꾼다.

    xt, eps_hat: (N, d), t: (N,), 0 < t <= 1
    반환: v (N, d)
    """
    xt, eps_hat = np.asarray(xt, float), np.asarray(eps_hat, float)
    t = np.asarray(t, float)[:, None]          # (N, 1)
    x1_hat = (xt - (1 - t) * eps_hat) / t      # E[x1 | x_t]
    return x1_hat - eps_hat                    # = (xt - eps_hat) / t
