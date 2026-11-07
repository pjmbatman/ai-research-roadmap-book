import numpy as np


def heat_residual(u, dx, dt, alpha):
    """격자 값 u (Nt, Nx)에서 열방정식 u_t = alpha u_xx의 내부 잔차 (Nt-2, Nx-2)를 반환한다.

    시간·공간 모두 중심차분을 쓴다.
    """
    u = np.asarray(u, dtype=float)
    u_t = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dt)                        # (Nt-2, Nx-2)
    u_xx = (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2      # (Nt-2, Nx-2)
    return u_t - alpha * u_xx
