import numpy as np


def classify(g, H, tol=1e-8):
    """gradient g (n,)와 Hessian H (n, n)로 점의 종류를 판정한다.

    반환: "not-stationary" | "min" | "max" | "saddle" | "inconclusive"
    """
    if np.linalg.norm(np.atleast_1d(g)) > tol:
        return "not-stationary"
    lam = np.linalg.eigvalsh(np.atleast_2d(H))     # 오름차순 고윳값
    has_pos, has_neg = lam[-1] > tol, lam[0] < -tol
    if has_pos and has_neg:
        return "saddle"
    if lam[0] > tol:
        return "min"
    if lam[-1] < -tol:
        return "max"
    return "inconclusive"
