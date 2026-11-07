import numpy as np

BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


def _adam_direction(g, m, v, t):
    m = BETA1 * m + (1 - BETA1) * g
    v = BETA2 * v + (1 - BETA2) * g * g
    u = (m / (1 - BETA1**t)) / (np.sqrt(v / (1 - BETA2**t)) + EPS)
    return u, m, v


def adam_l2_step(x, g, m, v, t, lr, wd):
    """L2 정규화: g + wd * x 로 Adam step. 반환 (x_new, m_new, v_new)."""
    x = np.asarray(x, dtype=float)
    u, m, v = _adam_direction(np.asarray(g, dtype=float) + wd * x, m, v, t)
    return x - lr * u, m, v


def adamw_step(x, g, m, v, t, lr, wd):
    """decoupled weight decay(AdamW). 반환 (x_new, m_new, v_new)."""
    x = np.asarray(x, dtype=float)
    u, m, v = _adam_direction(np.asarray(g, dtype=float), m, v, t)
    return x - lr * u - lr * wd * x, m, v
