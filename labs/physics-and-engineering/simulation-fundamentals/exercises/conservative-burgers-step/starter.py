import numpy as np


def burgers_fv_step(u, dx, dt):
    """보존형 유한체적법으로 u_t + (u^2/2)_x = 0을 한 step 진행한다(주기 경계, Rusanov 유속).

    u: (N,) 셀 평균, dx: 셀 길이, dt: 시간 간격
    반환: (N,) 한 step 뒤의 셀 평균
    """
    raise NotImplementedError("면 유속을 계산하고 셀 평균을 갱신하세요")
