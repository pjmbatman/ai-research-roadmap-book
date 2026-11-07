import numpy as np


def unbroadcast(grad, shape):
    """broadcast된 출력의 gradient를 원래 입력 shape으로 합해 줄인다.

    grad: 출력 쪽 gradient (출력 shape)
    shape: 입력의 원래 shape (tuple)
    반환: shape 모양의 gradient
    """
    raise NotImplementedError("앞에 붙은 축과 길이 1에서 늘어난 축을 합하세요")
