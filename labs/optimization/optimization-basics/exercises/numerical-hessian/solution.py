import numpy as np


def hessian_fd(f, x, h=1e-4):
    """중앙 차분으로 f의 Hessian을 근사한다.

    f: (n,) 배열을 받아 스칼라를 반환하는 함수
    x: (n,)
    반환: (n, n) 대칭행렬
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    E = np.eye(n) * h                       # (n, n) 좌표 방향 증분
    H = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = (f(x + E[i] + E[j]) - f(x + E[i] - E[j])
                       - f(x - E[i] + E[j]) + f(x - E[i] - E[j])) / (4 * h * h)
    return (H + H.T) / 2                    # 반올림 오차로 생긴 비대칭 제거


def is_convex_on(f, points, tol=1e-6):
    """points (M, n)의 모든 점에서 Hessian 최소 고윳값이 -tol 이상이면 True."""
    for p in np.atleast_2d(points):
        if np.linalg.eigvalsh(hessian_fd(f, p))[0] < -tol:
            return False
    return True
