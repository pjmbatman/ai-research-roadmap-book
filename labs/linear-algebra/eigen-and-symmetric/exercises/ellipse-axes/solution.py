import numpy as np


def ellipse_axes(S):
    """x^T S x = 1인 타원(체)의 반축 길이와 방향을 구한다.

    S: (n, n) 대칭 양의 정부호
    반환: lengths (n,) 내림차순, directions (n, n) 열 = 단위 방향
    S가 대칭이 아니거나 양의 정부호가 아니면 ValueError
    """
    S = np.asarray(S, dtype=float)
    if not np.allclose(S, S.T, atol=1e-10):
        raise ValueError("S가 대칭이 아닙니다")
    w, Q = np.linalg.eigh(S)            # 오름차순 고윳값 (n,), 정규직교 열 (n, n)
    if w[0] <= 0:
        raise ValueError("S가 양의 정부호가 아닙니다")
    return 1.0 / np.sqrt(w), Q          # 작은 고윳값 = 긴 반축이 먼저
