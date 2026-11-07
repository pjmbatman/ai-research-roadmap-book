import numpy as np


def adam_step(x, g, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam 한 step.

    x, g, m, v: (n,), t: 1부터 시작하는 step 번호
    반환: (x_new, m_new, v_new). m_new, v_new는 bias correction 전의 값.
    """
    g = np.asarray(g, dtype=float)
    m = beta1 * m + (1 - beta1) * g             # 1차 모멘트(gradient의 EMA)
    v = beta2 * v + (1 - beta2) * g * g         # 2차 모멘트(제곱 gradient의 EMA)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    x = np.asarray(x, dtype=float) - lr * m_hat / (np.sqrt(v_hat) + eps)
    return x, m, v
