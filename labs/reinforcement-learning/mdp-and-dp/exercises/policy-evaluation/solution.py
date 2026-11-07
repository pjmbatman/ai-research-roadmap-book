import numpy as np


def evaluate_policy(P, R, pi, gamma):
    """정책 pi의 상태 가치 V^pi를 Bellman 기대 방정식의 선형계로 정확히 푼다.

    P: (S, A, S) 전이확률 P[s, a, s'], R: (S, A) 기대 즉시 보상,
    pi: (S, A) 정책 pi(a|s) (행의 합 1), gamma: 할인율 (0 <= gamma < 1)
    반환: V (S,)
    """
    P, R, pi = map(np.asarray, (P, R, pi))
    S = P.shape[0]
    P_pi = np.einsum("sa,sat->st", pi, P)    # (S, S) 정책을 따를 때의 전이행렬
    r_pi = (pi * R).sum(axis=1)              # (S,)
    return np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
