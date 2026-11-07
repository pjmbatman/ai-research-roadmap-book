import numpy as np


def ridge_fit(X, y, lam):
    """절편 없는 ridge 회귀. X: (N, D), y: (N,). 반환: w (D,)"""
    raise NotImplementedError("(X^T X + lam I) w = X^T y를 푸세요")


def cv_mse(X, y, lam, k):
    """fold 배정 np.arange(N) % k로 k-fold CV 평균 MSE를 계산한다."""
    raise NotImplementedError("fold마다 학습·평가해 MSE를 평균하세요")


def nested_cv_mse(X, y, lambdas, k_outer=5, k_inner=5):
    """바깥 fold의 학습 부분 안에서만 lambda를 고르는 nested CV.

    반환: (바깥 fold MSE의 평균, 바깥 fold마다 고른 lambda 리스트)
    """
    raise NotImplementedError("바깥 fold마다 안쪽 CV로 lambda를 고르고 바깥 시험 fold에서 평가하세요")
