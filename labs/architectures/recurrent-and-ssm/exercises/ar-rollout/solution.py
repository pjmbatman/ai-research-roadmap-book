import numpy as np


def one_step(w, x):
    """참값 과거로 한 step씩 예측한다(teacher forcing).

    w: (p,), w[0]이 바로 전 시점의 계수.  x: (T,) 참값
    반환: (T,)  앞 p개는 x 그대로, t >= p에서 sum_i w[i-1] * x[t-i]
    """
    w, x = np.asarray(w, float), np.asarray(x, float)
    p = len(w)
    out = x.copy()
    for t in range(p, len(x)):
        out[t] = w @ x[t - p:t][::-1]          # 과거는 참값 x에서
    return out


def rollout(w, x_init, T):
    """처음 p개 참값에서 시작해 자기 예측을 과거로 넣으며 T개까지 이어 간다.

    w: (p,), x_init: (p,), T: 전체 길이
    반환: (T,)  앞 p개는 x_init
    """
    w = np.asarray(w, float)
    p = len(w)
    out = np.empty(T)
    out[:p] = x_init
    for t in range(p, T):
        out[t] = w @ out[t - p:t][::-1]        # 과거는 지금까지의 예측에서
    return out
