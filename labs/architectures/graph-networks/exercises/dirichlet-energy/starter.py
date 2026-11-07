import numpy as np


def dirichlet_energy(A, H):
    """이웃 노드 표현 사이의 차이를 모은 Dirichlet 에너지를 계산한다.

    A: (N, N) 대칭 인접행렬
    H: (N, D) 노드 표현
    반환: float, 0.5 * sum_ij A_ij ||h_i - h_j||^2
    """
    raise NotImplementedError("이웃 쌍마다 표현 차이의 제곱을 더하세요")
