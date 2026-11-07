import numpy as np


def burgers_fv_step(u, dx, dt):
    """보존형 유한체적법으로 u_t + (u^2/2)_x = 0을 한 step 진행한다(주기 경계, Rusanov 유속).

    u: (N,) 셀 평균, dx: 셀 길이, dt: 시간 간격
    반환: (N,) 한 step 뒤의 셀 평균
    """
    u = np.asarray(u, dtype=float)
    ur = np.roll(u, -1)                                         # (N,) 오른쪽 이웃 u_{j+1}
    a = np.maximum(np.abs(u), np.abs(ur))                       # (N,) 면에서의 최대 파동 속도
    F = 0.5 * (0.5 * u**2 + 0.5 * ur**2) - 0.5 * a * (ur - u)   # (N,) F_{j+1/2}
    return u - dt / dx * (F - np.roll(F, 1))
