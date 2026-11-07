import numpy as np


def mlp_grads(X, y, W1, b1, W2, b2):
    """2층 ReLU MLP + 평균 softmax cross-entropy의 파라미터 gradient.

    X: (N, D), y: (N,), W1: (H, D), b1: (H,), W2: (K, H), b2: (K,)
    반환: {"W1": (H, D), "b1": (H,), "W2": (K, H), "b2": (K,)}
    """
    raise NotImplementedError("forward를 계산한 뒤 출력 쪽부터 gradient를 구하세요")
