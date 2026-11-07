import numpy as np


def euler_with_sensitivity(r, K, y0, h, steps):
    """y' = r y (1 - y/K)를 explicit Euler로 적분하며 s = dy/dr도 함께 계산한다.

    반환: y (steps+1,), s (steps+1,). y[0] = y0, s[0] = 0.
    """
    y, s = np.empty(steps + 1), np.empty(steps + 1)
    y[0], s[0] = y0, 0.0
    for n in range(steps):
        f_y = r * (1 - 2 * y[n] / K)                    # ∂f/∂y
        f_r = y[n] * (1 - y[n] / K)                     # ∂f/∂r
        s[n + 1] = s[n] + h * (f_y * s[n] + f_r)        # 갱신 전 y[n]을 쓴다
        y[n + 1] = y[n] + h * r * y[n] * (1 - y[n] / K)
    return y, s
