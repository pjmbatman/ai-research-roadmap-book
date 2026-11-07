import numpy as np


def mlp_jacobian(W1, b1, W2, b2, x):
    """g(x) = W2 tanh(W1 x + b1) + b2의 Jacobian을 chain rule로 계산한다.

    W1: (k, n), b1: (k,), W2: (m, k), b2: (m,), x: (n,)
    반환: (m, n)
    """
    raise NotImplementedError("세 사상의 Jacobian을 구해 곱하세요")
