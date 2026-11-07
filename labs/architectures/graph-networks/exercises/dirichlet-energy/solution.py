import numpy as np


def dirichlet_energy(A, H):
    """이웃 노드 표현 사이의 차이를 모은 Dirichlet 에너지를 계산한다.

    A: (N, N) 대칭 인접행렬
    H: (N, D) 노드 표현
    반환: float, 0.5 * sum_ij A_ij ||h_i - h_j||^2
    """
    A = np.asarray(A, dtype=float)                      # (N, N)
    H = np.asarray(H, dtype=float)                      # (N, D)
    diff2 = ((H[:, None, :] - H[None, :, :]) ** 2).sum(-1)   # 모든 쌍의 거리 제곱 (N, N)
    return float(0.5 * (A * diff2).sum())
