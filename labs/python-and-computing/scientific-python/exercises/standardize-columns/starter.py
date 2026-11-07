import numpy as np


def standardize(X):
    """X의 각 열을 평균 0, 표준편차 1로 표준화한다. X는 바꾸지 않는다.

    X: (N, D)
    반환: Z (N, D) 실수, mu (D,), sigma (D,) — 상수 열의 sigma는 1
    """
    raise NotImplementedError("Z, mu, sigma를 계산하세요")
