import numpy as np


def flow_log_prob(x, s_fn, t_fn):
    """x1 = z1, x2 = z2 * exp(s(z1)) + t(z1), z ~ N(0, I)인 coupling flow에서 log p(x)를 반환한다.

    x: (N, 2), s_fn, t_fn: (N,) -> (N,)
    반환: (N,)
    """
    x = np.asarray(x, dtype=float)
    z1 = x[:, 0]
    s = s_fn(z1)
    z2 = (x[:, 1] - t_fn(z1)) * np.exp(-s)                 # 역변환
    log_base = -0.5 * (z1**2 + z2**2) - np.log(2 * np.pi)  # log N(z; 0, I)
    return log_base - s                                    # log|det dx/dz| = s(z1)
