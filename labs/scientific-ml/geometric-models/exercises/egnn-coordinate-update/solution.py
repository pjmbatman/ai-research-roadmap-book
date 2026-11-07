import numpy as np


def egnn_coords(x, phi):
    """EGNN 좌표 갱신 x_i + (1/(N-1)) sum_j (x_i - x_j) phi(||x_i - x_j||^2).

    x: (N, D) 좌표, phi: (N, N) 거리 제곱 → (N, N) 가중치 (원소별 함수)
    반환: x_new (N, D)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    diff = x[:, None, :] - x[None, :, :]          # (N, N, D) x_i - x_j: 회전에 등변, 평행이동에 불변
    d2 = np.sum(diff**2, axis=-1)                 # (N, N) 거리 제곱: 불변
    w = phi(d2) * (1.0 - np.eye(n))               # (N, N) j = i 항 제외
    return x + np.sum(diff * w[..., None], axis=1) / (n - 1)
