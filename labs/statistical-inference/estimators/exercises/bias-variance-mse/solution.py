import numpy as np


def bias_variance_mse(estimator, sampler, theta, n, reps=20000, seed=0):
    """같은 실험을 reps번 반복해 추정량의 (bias, variance, mse)를 반환한다.

    sampler(rng, size) -> (reps, n) 배열, estimator(X) -> (reps,) 배열.
    variance는 reps로 나눈다(np.var 기본값).
    """
    rng = np.random.default_rng(seed)
    X = sampler(rng, (reps, n))                      # (reps, n)
    est = np.asarray(estimator(X), dtype=float)      # (reps,)
    bias = float(est.mean() - theta)
    variance = float(est.var())
    mse = float(np.mean((est - theta) ** 2))
    return bias, variance, mse
