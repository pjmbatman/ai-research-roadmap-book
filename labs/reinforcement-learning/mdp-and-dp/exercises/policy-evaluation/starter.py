import numpy as np


def evaluate_policy(P, R, pi, gamma):
    """정책 pi의 상태 가치 V^pi를 Bellman 기대 방정식의 선형계로 정확히 푼다.

    P: (S, A, S) 전이확률 P[s, a, s'], R: (S, A) 기대 즉시 보상,
    pi: (S, A) 정책 pi(a|s) (행의 합 1), gamma: 할인율 (0 <= gamma < 1)
    반환: V (S,)
    """
    raise NotImplementedError("P_pi와 r_pi를 만들고 (I - gamma P_pi) V = r_pi를 푸세요")
