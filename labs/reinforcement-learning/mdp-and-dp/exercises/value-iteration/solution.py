import numpy as np


def value_iteration(P, R, gamma, tol=1e-10, max_iter=100_000):
    """가치 반복으로 최적 가치 V*와 탐욕적 결정적 정책을 구한다.

    P: (S, A, S) 전이확률, R: (S, A) 기대 즉시 보상, gamma: 할인율 (0 <= gamma < 1)
    반환: V (S,), pi (S,) 정수 배열 (상태마다 고른 행동 번호)
    """
    P, R = np.asarray(P, dtype=float), np.asarray(R, dtype=float)
    V = np.zeros(P.shape[0])
    for _ in range(max_iter):
        V_new = (R + gamma * P @ V).max(axis=1)      # Bellman 최적 연산자, (S,)
        done = np.abs(V_new - V).max() < tol
        V = V_new
        if done:
            break
    pi = (R + gamma * P @ V).argmax(axis=1)          # (S,)
    return V, pi
