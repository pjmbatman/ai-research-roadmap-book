import numpy as np


def ssm_scan(A_bar, B_bar, C, u):
    """재귀 x_k = A_bar x_{k-1} + B_bar u_k, y_k = C x_k (처음 상태 0).

    A_bar: (N, N), B_bar: (N, 1), C: (1, N), u: (L,)
    반환: y (L,)
    """
    A_bar, B_bar, C, u = map(np.asarray, (A_bar, B_bar, C, u))
    x = np.zeros((A_bar.shape[0], 1))               # (N, 1)
    y = np.empty(len(u))
    for k, u_k in enumerate(u):
        x = A_bar @ x + B_bar * u_k
        y[k] = (C @ x).item()
    return y


def ssm_kernel(A_bar, B_bar, C, L):
    """kernel K_j = C A_bar^j B_bar, j = 0..L-1.

    반환: K (L,)
    """
    A_bar, B_bar, C = map(np.asarray, (A_bar, B_bar, C))
    K = np.empty(L)
    v = B_bar.copy()                                # A_bar^j B_bar, (N, 1)
    for j in range(L):
        K[j] = (C @ v).item()
        v = A_bar @ v
    return K


def ssm_conv(A_bar, B_bar, C, u):
    """ssm_kernel과 인과 합성곱 y_k = sum_{j=0}^{k} K_j u_{k-j}로 출력을 계산한다.

    반환: y (L,)
    """
    u = np.asarray(u, dtype=float)
    K = ssm_kernel(A_bar, B_bar, C, len(u))
    return np.convolve(u, K)[: len(u)]              # 전체 합성곱의 앞 L개 = 인과 합성곱
