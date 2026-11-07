import numpy as np


def posterior(Phi, y, alpha, sigma):
    """베이지안 선형회귀의 가중치 사후분포 N(m, S)를 구한다.

    Phi: (N, D), y: (N,), alpha: 사전 정밀도, sigma: 관측 잡음 표준편차
    반환: m (D,), S (D, D)
    """
    Phi, y = np.asarray(Phi, dtype=float), np.asarray(y, dtype=float)
    D = Phi.shape[1]
    S = np.linalg.inv(alpha * np.eye(D) + Phi.T @ Phi / sigma**2)   # (D, D)
    m = S @ Phi.T @ y / sigma**2                                     # (D,)
    return m, S


def predictive(Phi_new, m, S, sigma):
    """새 입력의 특징 Phi_new (M, D)에서 예측분포의 평균 (M,)과 분산 (M,)을 반환한다."""
    Phi_new = np.asarray(Phi_new, dtype=float)
    mean = Phi_new @ m                                               # (M,)
    var = sigma**2 + np.sum((Phi_new @ S) * Phi_new, axis=1)         # (M,)
    return mean, var
