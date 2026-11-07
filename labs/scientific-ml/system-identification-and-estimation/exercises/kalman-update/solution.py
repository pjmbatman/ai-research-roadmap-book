import numpy as np


def kf_update(m, P, y, H, R):
    """칼만 필터의 갱신 단계.

    m: (n,) 예측 평균, P: (n, n) 예측 공분산
    y: (k,) 측정, H: (k, n) 관측 행렬, R: (k, k) 측정 잡음 공분산
    반환: (m_new (n,), P_new (n, n))
    """
    m, P, y, H, R = (np.asarray(a, dtype=float) for a in (m, P, y, H, R))
    S = H @ P @ H.T + R                     # (k, k) 혁신 공분산
    K = np.linalg.solve(S, H @ P).T         # (n, k) 칼만 이득 P H^T S^{-1}
    m_new = m + K @ (y - H @ m)
    P_new = P - K @ H @ P
    return m_new, 0.5 * (P_new + P_new.T)   # 반올림으로 깨진 대칭을 맞춘다
