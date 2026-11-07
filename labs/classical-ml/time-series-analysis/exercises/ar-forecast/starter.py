import numpy as np


def fit_ar(x, p):
    """절편 없는 AR(p) 계수를 최소제곱으로 추정한다. x: (T,). 반환: phi (p,)"""
    raise NotImplementedError("lag 행렬을 만들고 최소제곱을 푸세요")


def forecast_ar(x, phi, h):
    """x의 마지막 값까지 보고 h 시점 앞까지 재귀적으로 예측한다. 반환: (h,)"""
    raise NotImplementedError("예측값을 다음 예측의 입력으로 쓰세요")
