import numpy as np


def fit_power_law(N, L, floor=0.0):
    """log(L - floor) = log a + b log N을 최소제곱으로 맞춰 (a, b)를 반환한다."""
    raise NotImplementedError("로그–로그 직선을 맞추세요")


def fit_with_floor(N, L, floors):
    """후보 바닥마다 맞춘 뒤 원래 척도 제곱오차가 가장 작은 (a, b, c)를 반환한다."""
    raise NotImplementedError("바닥을 격자 탐색하세요")
