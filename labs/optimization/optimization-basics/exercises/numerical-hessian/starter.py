import numpy as np


def hessian_fd(f, x, h=1e-4):
    """중앙 차분으로 f의 Hessian을 근사한다.

    f: (n,) 배열을 받아 스칼라를 반환하는 함수
    x: (n,)
    반환: (n, n) 대칭행렬
    """
    raise NotImplementedError("중앙 차분으로 Hessian을 계산하세요")


def is_convex_on(f, points, tol=1e-6):
    """points (M, n)의 모든 점에서 Hessian 최소 고윳값이 -tol 이상이면 True."""
    raise NotImplementedError("각 점의 Hessian 최소 고윳값을 확인하세요")
