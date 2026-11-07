import numpy as np


def batchnorm_forward(x, gamma, beta, running_mean, running_var, training, momentum=0.1, eps=1e-5):
    """BatchNorm 한 번의 forward.

    x: (B, D), gamma·beta·running_mean·running_var: (D,)
    반환: (y (B, D), new_running_mean (D,), new_running_var (D,))
    train: batch 통계로 정규화하고 이동 평균을 갱신한다(분산 갱신에는 불편 분산).
    eval: running 통계로 정규화하고 통계는 그대로 돌려준다.
    """
    x = np.asarray(x, dtype=float)
    gamma, beta = np.asarray(gamma, dtype=float), np.asarray(beta, dtype=float)
    running_mean = np.asarray(running_mean, dtype=float)
    running_var = np.asarray(running_var, dtype=float)
    if training:
        B = x.shape[0]
        if B < 2:
            raise ValueError("train 모드 BatchNorm에는 표본이 2개 이상 필요하다")
        mu = x.mean(axis=0)                                  # (D,)
        var_biased = x.var(axis=0)                           # (D,) 정규화에 쓰는 편향 분산
        var_unbiased = var_biased * B / (B - 1)              # (D,) 이동 평균에 쓰는 불편 분산
        x_hat = (x - mu) / np.sqrt(var_biased + eps)
        new_mean = (1 - momentum) * running_mean + momentum * mu
        new_var = (1 - momentum) * running_var + momentum * var_unbiased
    else:
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        new_mean, new_var = running_mean.copy(), running_var.copy()
    return gamma * x_hat + beta, new_mean, new_var
