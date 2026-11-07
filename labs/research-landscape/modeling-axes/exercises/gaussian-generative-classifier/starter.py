import numpy as np


def fit_predict_proba(x_train, y_train, x_new):
    """공유 분산 가우시안 생성 분류기를 학습하고 x_new마다 p(y=1|x)를 반환한다.

    x_train: (N,), y_train: (N,) 0 또는 1, x_new: (M,)
    반환: (M,) 확률
    """
    raise NotImplementedError("사전확률·평균·공유 분산을 추정하고 Bayes 정리로 p(y=1|x)를 계산하세요")
