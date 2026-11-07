import numpy as np


def min_norm_solution(X, y):
    """X w = y를 만족하는 해 중 노름이 가장 작은 해 (p,). pinv를 쓰지 않는다."""
    raise NotImplementedError("X^T (X X^T)^{-1} y를 계산하세요")


def gd_limit(X, y, w0):
    """0.5‖Xw - y‖²에 w0에서 시작한 경사하강이 수렴하는 해 (p,)를 닫힌 형태로 계산한다."""
    raise NotImplementedError("경사하강의 도착점을 계산하세요")
