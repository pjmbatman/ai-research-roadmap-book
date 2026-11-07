import numpy as np


def kaplan_meier(time, event):
    """Kaplan–Meier 추정. time: (N,), event: (N,) bool. 반환: (t_j, S_j), 각각 (J,)"""
    raise NotImplementedError("고장 시각마다 d_j와 n_j를 세어 (1 - d_j/n_j)를 누적해 곱하세요")


def km_median(t_j, S_j):
    """S가 처음 0.5 이하가 되는 시각. 없으면 np.inf"""
    raise NotImplementedError("S_j <= 0.5인 첫 시각을 찾으세요")
