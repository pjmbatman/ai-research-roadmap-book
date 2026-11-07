import numpy as np


def crps_ensemble(samples, y):
    """표본으로 주어진 예측분포의 CRPS를 관측마다 계산한다.

    samples: (S, N) 관측마다 예측 표본 S개, y: (N,) 관측값
    반환: (N,) 관측마다의 CRPS
    """
    raise NotImplementedError("E|X - y| - E|X - X'| / 2 를 표본 평균으로 계산하세요")
