import numpy as np


def standardize_split(X_train, X_test):
    """train에서 열마다 평균·표준편차를 추정해 train과 test를 표준화한다.

    X_train: (N_train, D), X_test: (N_test, D)  — NaN 허용
    반환: Z_train (N_train, D), Z_test (N_test, D), mean (D,), std (D,)
    """
    raise NotImplementedError("train에서만 mean, std를 구해 두 배열에 적용하세요")
