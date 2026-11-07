import numpy as np


def conv_matrix(kernel, T):
    """길이 k(홀수)인 kernel의 원형 1차원 합성곱을 (T, T) 행렬로 반환한다.

    y_t = sum_j kernel[j] * x[(t + j - c) % T],  c = (k - 1) // 2
    kernel: (k,), T: 신호 길이
    반환: M (T, T), M @ x == y
    """
    kernel = np.asarray(kernel, dtype=float)
    c = (len(kernel) - 1) // 2
    M = np.zeros((T, T))
    for t in range(T):
        for j, w in enumerate(kernel):
            M[t, (t + j - c) % T] += w
    return M
