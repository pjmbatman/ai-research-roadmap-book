import numpy as np


def batchnorm_forward(x, gamma, beta, running_mean, running_var, training, momentum=0.1, eps=1e-5):
    """BatchNorm 한 번의 forward.

    x: (B, D), gamma·beta·running_mean·running_var: (D,)
    반환: (y (B, D), new_running_mean (D,), new_running_var (D,))
    train: batch 통계로 정규화하고 이동 평균을 갱신한다(분산 갱신에는 불편 분산).
    eval: running 통계로 정규화하고 통계는 그대로 돌려준다.
    """
    raise NotImplementedError("train/eval 두 경우를 구현하세요")
