import numpy as np


def conv2d(x, w, b, stride=1, padding=0):
    """2차원 합성곱(상호상관) forward.

    x: (C_in, H, W), w: (C_out, C_in, kH, kW), b: (C_out,)
    stride, padding: 높이·너비에 같은 값을 쓴다. padding은 0으로 채운다.
    반환: y (C_out, H_out, W_out),  H_out = (H + 2*padding - kH) // stride + 1
    """
    raise NotImplementedError("출력 shape을 계산하고 y를 채우세요")
