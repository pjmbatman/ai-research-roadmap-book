import numpy as np


def permutation_importance(predict, X, y, n_repeats=5, seed=0):
    """특징마다 값을 섞었을 때 늘어나는 MSE의 평균을 반환한다.

    predict: (N, p) → (N,) 함수, X: (N, p), y: (N,)
    반환: (p,) 중요도
    """
    raise NotImplementedError("기준 MSE → 열을 섞은 복사본의 MSE → 증가량의 평균")
