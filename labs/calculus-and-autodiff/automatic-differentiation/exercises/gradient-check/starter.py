import numpy as np


def grad_check(f, grad, x, eps=1e-6):
    """중앙차분과 비교한 성분별 상대오차의 최댓값.

    f: (n,) -> 스칼라, grad: (n,), x: (n,)
    반환: 스칼라 max_i |g_i - g_num_i| / max(|g_i| + |g_num_i|, 1e-12)
    """
    raise NotImplementedError("float64로 중앙차분 gradient를 계산하고 상대오차의 최댓값을 반환하세요")
