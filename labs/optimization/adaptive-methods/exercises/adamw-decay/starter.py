import numpy as np

BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


def adam_l2_step(x, g, m, v, t, lr, wd):
    """L2 정규화: g + wd * x 로 Adam step. 반환 (x_new, m_new, v_new)."""
    raise NotImplementedError("gradient에 wd * x를 더한 뒤 Adam step을 하세요")


def adamw_step(x, g, m, v, t, lr, wd):
    """decoupled weight decay(AdamW). 반환 (x_new, m_new, v_new)."""
    raise NotImplementedError("모멘트는 g로만 갱신하고, 감쇠 lr * wd * x를 따로 빼세요")
