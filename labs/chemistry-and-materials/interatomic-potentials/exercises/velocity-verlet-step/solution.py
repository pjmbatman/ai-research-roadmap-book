import numpy as np


def verlet_step(R, V, force_fn, dt):
    """velocity Verlet 한 스텝 (질량 1). 갱신된 (R_new, V_new)를 반환한다.

    R, V: (n, 3), force_fn: 위치 -> 힘 (n, 3), dt: 시간 간격
    """
    R = np.asarray(R, dtype=float)
    V = np.asarray(V, dtype=float)
    V_half = V + 0.5 * dt * force_fn(R)      # 반 스텝 속도 (현재 위치의 힘)
    R_new = R + dt * V_half                  # 위치 갱신
    V_new = V_half + 0.5 * dt * force_fn(R_new)   # 나머지 반 스텝 (갱신된 위치의 힘)
    return R_new, V_new
