import numpy as np


def kf_update(m, P, y, H, R):
    """칼만 필터의 갱신 단계.

    m: (n,) 예측 평균, P: (n, n) 예측 공분산
    y: (k,) 측정, H: (k, n) 관측 행렬, R: (k, k) 측정 잡음 공분산
    반환: (m_new (n,), P_new (n, n))
    """
    raise NotImplementedError("혁신 공분산 S, 칼만 이득 K를 계산해 평균과 공분산을 갱신하세요")
