import numpy as np


def posterior(Phi, y, alpha, sigma):
    """베이지안 선형회귀의 가중치 사후분포 N(m, S)를 구한다.

    Phi: (N, D), y: (N,), alpha: 사전 정밀도, sigma: 관측 잡음 표준편차
    반환: m (D,), S (D, D)
    """
    raise NotImplementedError("S와 m을 계산하세요")


def predictive(Phi_new, m, S, sigma):
    """새 입력의 특징 Phi_new (M, D)에서 예측분포의 평균 (M,)과 분산 (M,)을 반환한다."""
    raise NotImplementedError("예측 평균과 분산을 계산하세요")
