import numpy as np


def ssm_scan(A_bar, B_bar, C, u):
    """재귀 x_k = A_bar x_{k-1} + B_bar u_k, y_k = C x_k (처음 상태 0).

    A_bar: (N, N), B_bar: (N, 1), C: (1, N), u: (L,)
    반환: y (L,)
    """
    raise NotImplementedError("상태를 한 step씩 갱신하세요")


def ssm_kernel(A_bar, B_bar, C, L):
    """kernel K_j = C A_bar^j B_bar, j = 0..L-1.

    반환: K (L,)
    """
    raise NotImplementedError("A_bar^j B_bar를 한 step씩 갱신하세요")


def ssm_conv(A_bar, B_bar, C, u):
    """ssm_kernel과 인과 합성곱 y_k = sum_{j=0}^{k} K_j u_{k-j}로 출력을 계산한다.

    반환: y (L,)
    """
    raise NotImplementedError("kernel을 만들고 합성곱하세요")
