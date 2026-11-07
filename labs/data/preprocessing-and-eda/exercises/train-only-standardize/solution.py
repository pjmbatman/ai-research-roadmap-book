import numpy as np


def standardize_split(X_train, X_test):
    """train에서 열마다 평균·표준편차를 추정해 train과 test를 표준화한다.

    X_train: (N_train, D), X_test: (N_test, D)  — NaN 허용
    반환: Z_train (N_train, D), Z_test (N_test, D), mean (D,), std (D,)
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    mean = np.nanmean(X_train, axis=0)            # (D,)
    std = np.nanstd(X_train, axis=0)              # (D,) ddof=0
    std = np.where(std > 0, std, 1.0)             # 상수 열은 나누지 않는다
    return (X_train - mean) / std, (X_test - mean) / std, mean, std
