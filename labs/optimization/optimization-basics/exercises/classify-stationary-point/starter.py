import numpy as np


def classify(g, H, tol=1e-8):
    """gradient g (n,)와 Hessian H (n, n)로 점의 종류를 판정한다.

    반환: "not-stationary" | "min" | "max" | "saddle" | "inconclusive"
    """
    raise NotImplementedError("gradient 크기와 Hessian 고윳값으로 판정하세요")
