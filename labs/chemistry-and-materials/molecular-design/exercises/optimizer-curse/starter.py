import numpy as np


def selection_bias(pred, true, k):
    """예측 상위 k개 후보에서의 평균 낙관 편향 mean(pred - true)를 반환한다.

    pred: (N,) 예측 물성, true: (N,) 참 물성, k: 상위 개수
    """
    raise NotImplementedError("예측 상위 k개를 고르고 pred - true의 평균을 구하세요")
