import numpy as np


def adam_step(x, g, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam 한 step.

    x, g, m, v: (n,), t: 1부터 시작하는 step 번호
    반환: (x_new, m_new, v_new). m_new, v_new는 bias correction 전의 값.
    """
    raise NotImplementedError("모멘트를 갱신하고, bias correction 후 x를 갱신하세요")
