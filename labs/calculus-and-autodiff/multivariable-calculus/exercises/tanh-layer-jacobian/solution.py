import numpy as np


def mlp_jacobian(W1, b1, W2, b2, x):
    """g(x) = W2 tanh(W1 x + b1) + b2의 Jacobian을 chain rule로 계산한다.

    W1: (k, n), b1: (k,), W2: (m, k), b2: (m,), x: (n,)
    반환: (m, n)
    """
    W1, b1, W2, x = map(np.asarray, (W1, b1, W2, x))
    h = np.tanh(W1 @ x + b1)                  # (k,)
    J_tanh = np.diag(1 - h**2)                # (k, k)
    return W2 @ J_tanh @ W1                   # (m, k)(k, k)(k, n) = (m, n)
