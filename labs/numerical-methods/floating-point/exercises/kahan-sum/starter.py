import numpy as np


def kahan_sum(a):
    """1차원 float32 배열 a의 합을 Kahan 보정 합으로 계산한다(모든 중간값 float32).

    a: (N,) float32
    반환: float32 스칼라
    """
    raise NotImplementedError("보정 변수 c를 두고 덧셈마다 잃은 부분을 되돌려 주세요")
