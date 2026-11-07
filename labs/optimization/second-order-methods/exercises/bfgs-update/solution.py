import numpy as np


def bfgs_update(Hinv, s, y):
    """BFGS 역Hessian 근사의 갱신.

    Hinv: (n, n) 현재 역Hessian 근사, s: (n,) 파라미터 변화, y: (n,) gradient 변화
    반환: (n, n) 새 역Hessian 근사 (입력 Hinv는 바꾸지 않는다)
    """
    Hinv, s, y = np.asarray(Hinv, float), np.asarray(s, float), np.asarray(y, float)
    rho = 1.0 / (y @ s)
    V = np.eye(len(s)) - rho * np.outer(s, y)                # (n, n)
    return V @ Hinv @ V.T + rho * np.outer(s, s)
