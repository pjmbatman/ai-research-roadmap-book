import numpy as np


def fit_ridge(X, y, lam):
    """표준화한 특징에 Ridge 벌점을 걸고, 원래 단위의 (w, b)를 반환한다.

    X: (N, p), y: (N,), lam: 정규화 강도 λ ≥ 0
    반환: w (p,), b (스칼라) — 예측은 X @ w + b
    """
    raise NotImplementedError("표준화 → 중심화 → Ridge 해 → 원래 단위로 되돌리기")
