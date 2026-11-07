import numpy as np


def log_softmax(z, axis=-1):
    """axis 방향의 log-softmax를 overflow·underflow 없이 계산한다.

    z: 임의 shape의 배열(logits)
    반환: z와 같은 shape, log softmax(z)
    """
    z = np.asarray(z, dtype=float)
    m = np.max(z, axis=axis, keepdims=True)                       # 축마다 최댓값
    lse = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
    return z - lse
