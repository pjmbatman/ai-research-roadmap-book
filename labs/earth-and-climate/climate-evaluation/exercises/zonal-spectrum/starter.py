import numpy as np


def energy_ratio(fc, truth):
    """동서 방향 파수별 에너지 비 P_fc(k) / P_truth(k), k = 1..nlon/2, 와 비율이 처음 0.5 아래로 내려가는 k.

    fc, truth: (..., nlon)
    반환: ratio (nlon // 2,), k_half (int 또는 None)
    """
    raise NotImplementedError("파수별 에너지를 구해 비교하세요")
