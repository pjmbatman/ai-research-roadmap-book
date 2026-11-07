import numpy as np


def gd_quadratic(H, b, x0, lr, steps):
    """f(x) = x^T H x / 2 - b^T x 에 경사하강을 steps번 적용한 x를 반환한다.

    H: (n, n) 대칭 양의 정부호, b: (n,), x0: (n,)
    """
    H, b = np.asarray(H, dtype=float), np.asarray(b, dtype=float)
    x = np.array(x0, dtype=float)
    for _ in range(steps):
        x = x - lr * (H @ x - b)
    return x


def max_stable_lr(H):
    """GD가 수렴하는 학습률의 상한(스칼라)."""
    L = np.linalg.eigvalsh(np.asarray(H, dtype=float))[-1]   # 가장 큰 고윳값 = 최대 곡률
    return 2.0 / L
