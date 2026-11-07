import numpy as np


def naive_difference(y, t):
    """처치군 평균 - 대조군 평균. y: (N,), t: (N,) 0/1"""
    raise NotImplementedError("두 군의 평균 차이를 계산하세요")


def propensity(Z, t, iters=25):
    """절편 포함 로지스틱 회귀(Newton)로 P(T=1 | Z)를 추정한다. Z: (N, p). 반환: (N,)"""
    raise NotImplementedError("Newton 갱신을 iters번 반복하세요")


def ipw_effect(y, t, e):
    """정규화한 역확률 가중 추정량. e: 성향점수 (N,)"""
    raise NotImplementedError("두 군의 가중 평균 차이를 계산하세요")
