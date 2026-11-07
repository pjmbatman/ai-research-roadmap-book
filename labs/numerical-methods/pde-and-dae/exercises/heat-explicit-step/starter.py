import numpy as np


def ftcs_step(u, r):
    """열방정식 FTCS 한 step. 양 끝(Dirichlet)은 고정한다.

    u: shape (N+1,) 현재 온도 분포
    r: α Δt / Δx²
    반환: shape (N+1,) 새 배열 (입력 u는 바꾸지 않는다)
    """
    raise NotImplementedError("내부점 u[1:-1]을 이웃 u[2:], u[:-2]로 갱신하세요")


def max_stable_dt(dx, alpha):
    """FTCS가 안정한 가장 큰 Δt (r = 1/2)."""
    raise NotImplementedError("r = α Δt / Δx² ≤ 1/2를 Δt에 대해 푸세요")
