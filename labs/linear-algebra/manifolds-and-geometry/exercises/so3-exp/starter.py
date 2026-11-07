import numpy as np


def so3_exp(omega):
    """축-각 벡터 omega (3,) [rad]를 회전 행렬 R (3, 3)으로 보낸다. R = exp(hat(omega)).

    omega = 0과 아주 작은 각에서도 NaN 없이 동작해야 한다.
    """
    raise NotImplementedError("Rodrigues 공식을 구현하세요")
