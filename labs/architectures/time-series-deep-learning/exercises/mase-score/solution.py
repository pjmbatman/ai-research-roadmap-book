import numpy as np


def mase(y_true, y_pred, y_train, m=1):
    """mean absolute scaled error.

    y_true, y_pred: 같은 shape의 정답과 예측 (평가 구간)
    y_train: (n_train,) 학습 구간 시계열, 척도 계산에만 쓴다
    m: 계절 주기
    반환: float. 척도가 0이면 ValueError
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)              # (n_train,)
    scale = np.abs(y_train[m:] - y_train[:-m]).mean()       # 학습 구간 seasonal naive 1-step MAE
    if scale == 0:
        raise ValueError("학습 구간의 seasonal naive 오차가 0이라 MASE를 정의할 수 없습니다")
    return float(np.abs(y_true - y_pred).mean() / scale)
