import numpy as np


def permutation_importance(predict, X, y, cols, n_repeats=10, seed=0):
    """열 cols를 같은 순열로 함께 섞었을 때 평균 제곱오차가 늘어나는 양의 평균.

    predict: (N, D) -> (N,), X: (N, D), y: (N,), cols: 열 번호 리스트
    """
    raise NotImplementedError("기준 MSE를 구하고, 복사본의 열을 섞어 MSE 증가량을 평균 내세요")
