import numpy as np


def conv2d(x, w, b, stride=1, padding=0):
    """2차원 합성곱(상호상관) forward.

    x: (C_in, H, W), w: (C_out, C_in, kH, kW), b: (C_out,)
    stride, padding: 높이·너비에 같은 값을 쓴다. padding은 0으로 채운다.
    반환: y (C_out, H_out, W_out),  H_out = (H + 2*padding - kH) // stride + 1
    """
    x, w, b = np.asarray(x, float), np.asarray(w, float), np.asarray(b, float)
    C_out, C_in, kH, kW = w.shape
    xp = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))   # (C_in, H+2p, W+2p)
    H_out = (xp.shape[1] - kH) // stride + 1
    W_out = (xp.shape[2] - kW) // stride + 1
    y = np.empty((C_out, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            patch = xp[:, i * stride:i * stride + kH, j * stride:j * stride + kW]   # (C_in, kH, kW)
            y[:, i, j] = np.einsum("chw,ochw->o", patch, w) + b
    return y
