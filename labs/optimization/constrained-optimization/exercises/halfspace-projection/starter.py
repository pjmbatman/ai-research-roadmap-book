import numpy as np


def project_halfspace(p, a, c):
    """min ||x - p||^2  s.t.  a^T x <= c 의 해와 KKT 승수.

    p: (n,), a: (n,) (0이 아닌 벡터), c: 스칼라
    반환: x (n,), mu (스칼라, 0 이상)
    """
    raise NotImplementedError("제약이 활성인 경우와 비활성인 경우로 나누어 x와 mu를 구하세요")
