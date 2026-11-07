import numpy as np


def verlet_step(R, V, force_fn, dt):
    """velocity Verlet 한 스텝 (질량 1). 갱신된 (R_new, V_new)를 반환한다.

    R, V: (n, 3), force_fn: 위치 -> 힘 (n, 3), dt: 시간 간격
    """
    raise NotImplementedError("반 스텝 속도, 위치 갱신, 나머지 반 스텝 속도를 차례로 계산하세요")
