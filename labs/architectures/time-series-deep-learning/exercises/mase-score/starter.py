import numpy as np


def mase(y_true, y_pred, y_train, m=1):
    """mean absolute scaled error.

    y_true, y_pred: 같은 shape의 정답과 예측 (평가 구간)
    y_train: (n_train,) 학습 구간 시계열, 척도 계산에만 쓴다
    m: 계절 주기
    반환: float. 척도가 0이면 ValueError
    """
    raise NotImplementedError("학습 구간의 seasonal naive MAE로 예측 MAE를 나누세요")
