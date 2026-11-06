import numpy as np


def breakpoints(c, w, b, tol=1e-9):
    """f(x) = sum_j c_j * ReLU(w_j * x + b_j)의 기울기가 바뀌는 점을 오름차순으로 반환한다.

    c, w, b: 길이 n인 배열
    반환: 1차원 NumPy 배열
    """
    raise NotImplementedError("꺾이는 점을 찾으세요")
