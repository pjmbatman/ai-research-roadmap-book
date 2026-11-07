import numpy as np


def flow_log_prob(x, s_fn, t_fn):
    """x1 = z1, x2 = z2 * exp(s(z1)) + t(z1), z ~ N(0, I)인 coupling flow에서 log p(x)를 반환한다.

    x: (N, 2), s_fn, t_fn: (N,) -> (N,)
    반환: (N,)
    """
    raise NotImplementedError("역변환, log-det, 변수변환 공식을 차례로 적용하세요")
