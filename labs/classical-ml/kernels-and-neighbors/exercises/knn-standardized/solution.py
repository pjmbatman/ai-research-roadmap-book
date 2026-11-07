import numpy as np


def knn_predict(X_train, y_train, X_test, k):
    """학습 데이터의 통계량으로 표준화한 공간에서 kNN 회귀 예측을 한다.

    X_train: (N, p), y_train: (N,), X_test: (M, p), k: 이웃 수
    반환: (M,) 예측
    """
    X_train = np.asarray(X_train, dtype=float)                  # (N, p)
    X_test = np.asarray(X_test, dtype=float)                    # (M, p)
    y_train = np.asarray(y_train, dtype=float)                  # (N,)
    mu, s = X_train.mean(axis=0), X_train.std(axis=0)           # 학습 데이터의 통계량만 쓴다
    A, B = (X_train - mu) / s, (X_test - mu) / s
    dist = np.linalg.norm(B[:, None, :] - A[None, :, :], axis=2)   # (M, N)
    nearest = np.argsort(dist, axis=1)[:, :k]                   # (M, k)
    return y_train[nearest].mean(axis=1)
