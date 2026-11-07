import numpy as np


def central_diff_matrix(N, L):
    """주기 격자 x_i = i L / N (i = 0..N-1) 위 중심 차분 미분 행렬 (N, N).

    (D u)_i = (u_{i+1} - u_{i-1}) / (2h),  h = L / N,  인덱스는 N을 법으로 돈다.
    """
    h = L / N
    I = np.eye(N)
    plus = np.roll(I, 1, axis=1)      # (plus @ u)_i = u_{i+1}
    minus = np.roll(I, -1, axis=1)    # (minus @ u)_i = u_{i-1}
    return (plus - minus) / (2 * h)
