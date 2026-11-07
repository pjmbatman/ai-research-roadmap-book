import numpy as np


def best_split(x, y):
    """특징 하나 x (N,)와 출력 y (N,)에서 제곱오차 합을 최소화하는 문턱을 찾는다.

    반환: (threshold, sse) — x <= threshold가 왼쪽
    """
    raise NotImplementedError("후보 문턱(서로 다른 값의 중점)마다 SSE를 계산하세요")
