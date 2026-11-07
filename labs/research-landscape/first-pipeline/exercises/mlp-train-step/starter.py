import numpy as np


def train_step(params, X, y, lr):
    """tanh 은닉층 하나의 MLP를 MSE로 한 step 학습한다.

    params: {"W1": (D, H), "b1": (H,), "W2": (H, 1), "b2": (1,)}
    X: (N, D), y: (N, 1), lr: 학습률
    반환: (new_params, loss) — new_params는 새 dict, loss는 갱신 전 파라미터에서의 MSE
    """
    raise NotImplementedError("forward → loss → backward → update를 쓰세요")
