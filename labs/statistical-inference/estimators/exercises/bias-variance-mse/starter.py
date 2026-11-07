import numpy as np


def bias_variance_mse(estimator, sampler, theta, n, reps=20000, seed=0):
    """같은 실험을 reps번 반복해 추정량의 (bias, variance, mse)를 반환한다.

    sampler(rng, size) -> (reps, n) 배열, estimator(X) -> (reps,) 배열.
    variance는 reps로 나눈다(np.var 기본값).
    """
    raise NotImplementedError("반복 표본을 뽑고 추정값의 bias·variance·MSE를 계산한다")
