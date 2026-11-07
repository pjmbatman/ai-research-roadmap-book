import numpy as np


def train_linear(X, y, micro_batch, accum, lr, epochs, seed):
    """gradient accumulation이 있는 선형회귀 SGD.

    X: (N, D), y: (N,)
    반환: (w (D,), b float, n_steps int)
    """
    raise NotImplementedError("micro-batch gradient를 표본 수로 가중해 모은 뒤 accum개마다 갱신하세요")
