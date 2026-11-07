import numpy as np


def pod_basis(X, energy=0.999):
    """스냅샷 행렬 X (M, N)의 POD 기저를 구한다.

    반환: mean (N,), modes (r, N) — 누적 에너지 비율이 energy 이상이 되는 가장 작은 r개 모드
    """
    raise NotImplementedError("평균을 빼고 SVD로 모드를 구하세요")
