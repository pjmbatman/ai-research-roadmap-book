import numpy as np


def quantize(w, axis=None):
    """대칭 int8 양자화.

    w: 실수 배열. axis=None이면 텐서 전체에 scale 하나, axis=0이면 2차원 w의 행(출력 채널)마다 scale 하나.
    반환: (q, scale)
        q: w와 shape이 같은 int8 배열, 값은 -127..127
        scale: axis=None이면 float, axis=0이면 shape (행 수, 1)인 배열
    절댓값의 최댓값이 0인 묶음(모두 0)은 scale을 1.0으로 둔다.
    """
    w = np.asarray(w, dtype=np.float64)
    if axis is None:
        amax = np.abs(w).max()
        scale = float(amax / 127) if amax > 0 else 1.0
    else:
        amax = np.abs(w).max(axis=1, keepdims=True)                 # (행 수, 1)
        scale = np.where(amax > 0, amax / 127, 1.0)
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize(q, scale):
    """q * scale을 실수 배열로 돌려준다. scale은 quantize가 돌려준 그대로 받는다."""
    return np.asarray(q, dtype=np.float64) * scale
