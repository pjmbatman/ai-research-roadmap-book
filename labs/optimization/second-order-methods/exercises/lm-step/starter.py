import numpy as np


def lm_step(J, r, lam):
    """Levenberg-Marquardt step.

    J: (N, p) 잔차의 Jacobian, r: (N,) 잔차, lam: 감쇠 계수(0 이상)
    반환: step (p,)  — (J^T J + lam I) step = -J^T r 의 해
    """
    raise NotImplementedError("감쇠를 더한 정규방정식을 풀어 step을 반환하세요")
