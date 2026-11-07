import numpy as np


def jacobian(f, x, eps=1e-6):
    """중앙차분으로 f의 Jacobian을 근사한다.

    f: (n,) -> (m,) 또는 스칼라
    x: (n,)
    반환: (m, n). f가 스칼라를 돌려주면 (1, n)
    """
    raise NotImplementedError("열마다 중앙차분을 계산해 (m, n) 행렬을 만드세요")
