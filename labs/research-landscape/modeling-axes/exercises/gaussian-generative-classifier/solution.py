import numpy as np


def fit_predict_proba(x_train, y_train, x_new):
    """공유 분산 가우시안 생성 분류기를 학습하고 x_new마다 p(y=1|x)를 반환한다.

    x_train: (N,), y_train: (N,) 0 또는 1, x_new: (M,)
    반환: (M,) 확률
    """
    x = np.asarray(x_train, dtype=float)               # (N,)
    y = np.asarray(y_train).astype(int)                # (N,)
    x_new = np.asarray(x_new, dtype=float)             # (M,)

    pi = np.array([np.mean(y == 0), np.mean(y == 1)])  # (2,) 사전확률
    mu = np.array([x[y == 0].mean(), x[y == 1].mean()])   # (2,) 클래스 평균
    var = np.mean((x - mu[y]) ** 2)                    # 공유 분산(MLE)

    # 로그 p(x|y=k) + 로그 p(y=k). 공통 상수 -0.5 log(2 pi var)는 약분된다.
    log_joint = np.log(pi)[None, :] - (x_new[:, None] - mu[None, :]) ** 2 / (2 * var)   # (M, 2)
    logit = log_joint[:, 1] - log_joint[:, 0]          # (M,)
    return 0.5 * (1.0 + np.tanh(logit / 2))            # sigmoid(logit), 큰 |logit|에서도 overflow 없음
