import numpy as np


def stlsq(Theta, dX, thresh, iters=10):
    """순차 임계 최소제곱으로 dX ≈ Theta @ Xi의 희소한 Xi를 구한다.

    Theta: (T, p) 후보 함수 라이브러리, dX: (T, d) 시간미분
    thresh: 이보다 절댓값이 작은 계수를 0으로 만든다
    반환: Xi (p, d)
    """
    raise NotImplementedError("최소제곱 → 임계 → 남은 항으로 다시 최소제곱을 반복하세요")
