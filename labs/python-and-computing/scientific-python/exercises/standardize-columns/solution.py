import numpy as np


def standardize(X):
    """X의 각 열을 평균 0, 표준편차 1로 표준화한다. X는 바꾸지 않는다.

    X: (N, D)
    반환: Z (N, D) 실수, mu (D,), sigma (D,) — 상수 열의 sigma는 1
    """
    X = np.asarray(X, dtype=float)           # 정수 입력도 실수로 (새 배열이거나 원본의 읽기 전용 사용)
    mu = X.mean(axis=0)                      # (D,)
    sigma = X.std(axis=0)                    # (D,)
    sigma = np.where(sigma > 0, sigma, 1.0)  # 상수 열은 1로
    Z = (X - mu) / sigma                     # (N, D) - (D,) → (N, D), 새 배열
    return Z, mu, sigma
