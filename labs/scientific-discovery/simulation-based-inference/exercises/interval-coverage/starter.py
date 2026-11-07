import numpy as np


def coverage(samples, theta_true, level=0.9):
    """사후 표본으로 만든 중앙 level 구간이 참값을 포함하는 비율.

    samples: (M, L) 시험 데이터 M개 각각의 사후 표본 L개
    theta_true: (M,) 각 시험 데이터를 만든 참값
    level: 구간 수준 (예: 0.9)
    반환: 경험적 coverage (0과 1 사이의 실수)
    """
    raise NotImplementedError("행마다 중앙 구간을 만들고 참값이 들어가는 비율을 계산하세요")
