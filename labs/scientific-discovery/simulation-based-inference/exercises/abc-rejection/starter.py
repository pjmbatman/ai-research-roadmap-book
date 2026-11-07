import numpy as np


def abc_accept(theta_sim, S_sim, s_obs, q):
    """ABC rejection: 관측 요약 통계에 가장 가까운 비율 q의 시뮬레이션을 받아들인다.

    theta_sim: (N,) 사전분포에서 뽑은 파라미터
    S_sim: (N, k) 각 시뮬레이션의 요약 통계
    s_obs: (k,) 관측 데이터의 요약 통계
    q: 수용 비율 (0 < q <= 1)
    반환: 받아들인 파라미터 (약 q*N개,)
    """
    raise NotImplementedError("표준화한 거리를 계산하고 q 분위수 이하를 받아들이세요")
