import numpy as np


def knn_predict(X_train, y_train, X_test, k):
    """학습 데이터의 통계량으로 표준화한 공간에서 kNN 회귀 예측을 한다.

    X_train: (N, p), y_train: (N,), X_test: (M, p), k: 이웃 수
    반환: (M,) 예측
    """
    raise NotImplementedError("표준화 → 거리 → 가장 가까운 k개의 평균")
