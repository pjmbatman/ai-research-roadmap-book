import numpy as np


def one_step(w, x):
    """참값 과거로 한 step씩 예측한다(teacher forcing).

    w: (p,), w[0]이 바로 전 시점의 계수.  x: (T,) 참값
    반환: (T,)  앞 p개는 x 그대로, t >= p에서 sum_i w[i-1] * x[t-i]
    """
    raise NotImplementedError("참값 과거로 예측하세요")


def rollout(w, x_init, T):
    """처음 p개 참값에서 시작해 자기 예측을 과거로 넣으며 T개까지 이어 간다.

    w: (p,), x_init: (p,), T: 전체 길이
    반환: (T,)  앞 p개는 x_init
    """
    raise NotImplementedError("자기 예측을 과거로 넣으며 이어 가세요")
