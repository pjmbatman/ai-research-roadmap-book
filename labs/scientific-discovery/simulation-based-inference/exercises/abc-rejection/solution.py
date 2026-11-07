import numpy as np


def abc_accept(theta_sim, S_sim, s_obs, q):
    """ABC rejection: 관측 요약 통계에 가장 가까운 비율 q의 시뮬레이션을 받아들인다.

    theta_sim: (N,) 사전분포에서 뽑은 파라미터
    S_sim: (N, k) 각 시뮬레이션의 요약 통계
    s_obs: (k,) 관측 데이터의 요약 통계
    q: 수용 비율 (0 < q <= 1)
    반환: 받아들인 파라미터 (약 q*N개,)
    """
    theta_sim = np.asarray(theta_sim, dtype=float)
    S_sim = np.asarray(S_sim, dtype=float).reshape(len(theta_sim), -1)   # (N, k)
    s_obs = np.atleast_1d(np.asarray(s_obs, dtype=float))                # (k,)
    scale = S_sim.std(axis=0)                                            # (k,) 통계별 척도
    d = np.sqrt((((S_sim - s_obs) / scale) ** 2).sum(axis=1))            # (N,) 표준화 거리
    return theta_sim[d <= np.quantile(d, q)]
