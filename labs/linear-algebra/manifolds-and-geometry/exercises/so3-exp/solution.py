import numpy as np


def hat(w):
    """(3,) → (3, 3) 반대칭 행렬. hat(w) @ x = cross(w, x)."""
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def so3_exp(omega):
    """축-각 벡터 omega (3,) [rad]를 회전 행렬 R (3, 3)으로 보낸다. R = exp(hat(omega)).

    omega = 0과 아주 작은 각에서도 NaN 없이 동작해야 한다.
    """
    w = np.asarray(omega, dtype=float)
    th = np.linalg.norm(w)
    W = hat(w)                                  # (3, 3)
    if th < 1e-6:                               # Taylor 전개: sin θ/θ ≈ 1 - θ²/6, (1 - cos θ)/θ² ≈ 1/2 - θ²/24
        a = 1.0 - th**2 / 6.0
        b = 0.5 - th**2 / 24.0
    else:
        a = np.sin(th) / th
        b = (1.0 - np.cos(th)) / th**2
    return np.eye(3) + a * W + b * W @ W
