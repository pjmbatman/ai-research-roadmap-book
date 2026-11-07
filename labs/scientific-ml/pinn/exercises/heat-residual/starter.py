import numpy as np


def heat_residual(u, dx, dt, alpha):
    """격자 값 u (Nt, Nx)에서 열방정식 u_t = alpha u_xx의 내부 잔차 (Nt-2, Nx-2)를 반환한다.

    시간·공간 모두 중심차분을 쓴다.
    """
    raise NotImplementedError("시간 중심차분과 공간 2계 중심차분으로 잔차를 계산하세요")
