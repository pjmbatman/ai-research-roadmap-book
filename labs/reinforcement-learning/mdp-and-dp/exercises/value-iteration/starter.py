import numpy as np


def value_iteration(P, R, gamma, tol=1e-10, max_iter=100_000):
    """가치 반복으로 최적 가치 V*와 탐욕적 결정적 정책을 구한다.

    P: (S, A, S) 전이확률, R: (S, A) 기대 즉시 보상, gamma: 할인율 (0 <= gamma < 1)
    반환: V (S,), pi (S,) 정수 배열 (상태마다 고른 행동 번호)
    """
    raise NotImplementedError("V <- max_a (R + gamma P V)를 수렴할 때까지 반복하고 argmax로 정책을 구하세요")
