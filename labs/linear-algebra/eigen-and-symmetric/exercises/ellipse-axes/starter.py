import numpy as np


def ellipse_axes(S):
    """x^T S x = 1인 타원(체)의 반축 길이와 방향을 구한다.

    S: (n, n) 대칭 양의 정부호
    반환: lengths (n,) 내림차순, directions (n, n) 열 = 단위 방향
    S가 대칭이 아니거나 양의 정부호가 아니면 ValueError
    """
    raise NotImplementedError("eigh로 고유분해하고 반축 길이 1/sqrt(lambda)를 구하세요")
