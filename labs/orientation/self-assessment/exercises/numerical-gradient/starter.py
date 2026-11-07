import numpy as np


def numerical_grad(f, w, h=1e-5):
    """중심 차분으로 f의 gradient를 근사한다.

    f: (D,) 배열 → 스칼라
    w: (D,) 배열 (바꾸지 않는다)
    반환: (D,) 배열, i번째 성분은 (f(w + h e_i) - f(w - h e_i)) / (2h)
    """
    raise NotImplementedError("좌표마다 중심 차분을 계산하세요")
