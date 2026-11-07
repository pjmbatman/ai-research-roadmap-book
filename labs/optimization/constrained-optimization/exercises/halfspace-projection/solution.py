import numpy as np


def project_halfspace(p, a, c):
    """min ||x - p||^2  s.t.  a^T x <= c 의 해와 KKT 승수.

    p: (n,), a: (n,) (0이 아닌 벡터), c: 스칼라
    반환: x (n,), mu (스칼라, 0 이상)
    """
    p, a = np.asarray(p, dtype=float), np.asarray(a, dtype=float)
    violation = a @ p - c
    if violation <= 0:                       # p가 이미 feasible: 제약 비활성, mu = 0
        return p.copy(), 0.0
    mu = 2 * violation / (a @ a)             # x = p - (mu/2) a 를 a^T x = c 에 대입
    return p - mu / 2 * a, float(mu)
