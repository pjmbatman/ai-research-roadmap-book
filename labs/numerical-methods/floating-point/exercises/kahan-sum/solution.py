import numpy as np


def kahan_sum(a):
    """1차원 float32 배열 a의 합을 Kahan 보정 합으로 계산한다(모든 중간값 float32).

    a: (N,) float32
    반환: float32 스칼라
    """
    s, c = np.float32(0), np.float32(0)
    for v in np.asarray(a, dtype=np.float32):
        y = v - c              # 지난번에 잃은 부분을 보정
        t = s + y
        c = (t - s) - y        # 이번 덧셈에서 잃은 부분
        s = t
    return s
