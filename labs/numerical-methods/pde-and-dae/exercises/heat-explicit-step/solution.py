import numpy as np


def ftcs_step(u, r):
    """열방정식 FTCS 한 step. 양 끝(Dirichlet)은 고정한다.

    u: shape (N+1,) 현재 온도 분포
    r: α Δt / Δx²
    반환: shape (N+1,) 새 배열 (입력 u는 바꾸지 않는다)
    """
    u = np.asarray(u, dtype=float)
    new = u.copy()                                            # (N+1,) 양 끝은 그대로
    new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
    return new


def max_stable_dt(dx, alpha):
    """FTCS가 안정한 가장 큰 Δt (r = 1/2)."""
    return dx**2 / (2 * alpha)
