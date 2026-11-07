import numpy as np


def posterior(x0, xt, t, betas):
    """q(x_{t-1} | x_t, x_0)의 (평균, 분산)을 반환한다. t는 1부터 센다(betas[t - 1] = beta_t)."""
    betas = np.asarray(betas, dtype=float)
    x0, xt = np.asarray(x0, dtype=float), np.asarray(xt, dtype=float)
    abar = np.cumprod(1 - betas)
    a_t = abar[t - 1]
    a_prev = abar[t - 2] if t > 1 else 1.0          # abar_0 = 1
    b = betas[t - 1]
    var = (1 - a_prev) / (1 - a_t) * b
    mean = np.sqrt(a_prev) * b / (1 - a_t) * x0 + np.sqrt(1 - b) * (1 - a_prev) / (1 - a_t) * xt
    return mean, np.broadcast_to(var, np.shape(mean)).astype(float)
