import numpy as np


def stlsq(Theta, dX, thr=0.05, iters=10):
    """순차적 임계 최소제곱으로 희소한 계수 행렬을 찾는다.

    Theta: (T, p) 후보 함수 사전
    dX: (T, n) 상태의 시간 미분
    thr: 임계값. 절댓값이 이보다 작은 계수는 0으로 만든다
    iters: 자르기-다시 맞추기 반복 횟수
    반환: Xi (p, n)
    """
    raise NotImplementedError("최소제곱 → 작은 계수 자르기 → 남은 항으로 다시 최소제곱을 반복하세요")
