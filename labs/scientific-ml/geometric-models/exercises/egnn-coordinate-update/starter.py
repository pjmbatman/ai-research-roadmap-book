import numpy as np


def egnn_coords(x, phi):
    """EGNN 좌표 갱신 x_i + (1/(N-1)) sum_j (x_i - x_j) phi(||x_i - x_j||^2).

    x: (N, D) 좌표, phi: (N, N) 거리 제곱 → (N, N) 가중치 (원소별 함수)
    반환: x_new (N, D)
    """
    raise NotImplementedError("상대 벡터와 거리 제곱을 계산해 좌표를 갱신하세요")
