import numpy as np


def rk4_step(f, t, y, h):
    """고전적 RK4로 y' = f(t, y)를 한 step 전진한다.

    f: f(t, y) -> (d,) 배열
    t: 현재 시각 (float)
    y: 현재 상태, shape (d,) (리스트도 허용)
    h: step 크기
    반환: 다음 상태 y_{n+1}, shape (d,)
    """
    y = np.asarray(y, dtype=float)                      # (d,) 새 배열이므로 입력을 바꾸지 않는다
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + h / 2, y + h / 2 * k1), dtype=float)
    k3 = np.asarray(f(t + h / 2, y + h / 2 * k2), dtype=float)
    k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
