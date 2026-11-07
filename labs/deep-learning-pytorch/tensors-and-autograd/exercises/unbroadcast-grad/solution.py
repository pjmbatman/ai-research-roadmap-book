import numpy as np


def unbroadcast(grad, shape):
    """broadcast된 출력의 gradient를 원래 입력 shape으로 합해 줄인다.

    grad: 출력 쪽 gradient (출력 shape)
    shape: 입력의 원래 shape (tuple)
    반환: shape 모양의 gradient
    """
    grad = np.asarray(grad, dtype=float)
    shape = tuple(shape)
    extra = grad.ndim - len(shape)
    if extra > 0:                                   # 규칙 1: 앞에 붙은 축은 합해서 없앤다
        grad = grad.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if axes:                                        # 규칙 2: 길이 1에서 늘어난 축은 합하고 남긴다
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)
