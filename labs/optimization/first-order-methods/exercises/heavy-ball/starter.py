import numpy as np


def heavy_ball(grad, x0, lr, beta, steps):
    """Heavy ball momentum의 반복점들을 반환한다.

    grad: x (n,) -> gradient (n,)
    반환: (steps + 1, n) 배열. 첫 행은 x0.
    """
    raise NotImplementedError("v <- beta v - lr grad(x), x <- x + v 를 반복하세요")
