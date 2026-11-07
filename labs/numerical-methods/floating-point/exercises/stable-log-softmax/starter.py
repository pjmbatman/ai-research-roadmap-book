import numpy as np


def log_softmax(z, axis=-1):
    """axis 방향의 log-softmax를 overflow·underflow 없이 계산한다.

    z: 임의 shape의 배열(logits)
    반환: z와 같은 shape, log softmax(z)
    """
    raise NotImplementedError("최댓값을 빼는 log-sum-exp로 구현하세요")
