import numpy as np


def composite_simpson(f, a, b, n):
    """합성 Simpson 규칙으로 ∫_a^b f(x) dx를 근사한다.

    f: 배열을 받아 같은 shape의 배열을 돌려주는 함수
    a, b: 적분 구간
    n: 작은 구간의 수 (2 이상의 짝수). 아니면 ValueError
    반환: float
    """
    raise NotImplementedError("노드 x_i와 가중치 [1, 4, 2, ..., 4, 1]로 합을 계산하세요")
