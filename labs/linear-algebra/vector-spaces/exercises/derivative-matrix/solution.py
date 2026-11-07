import numpy as np


def derivative_matrix(n):
    """P_n 위의 미분 연산자를 단항식 basis (1, x, ..., x^n)로 표현한 행렬.

    n: 다항식의 최고 차수 (n >= 0)
    반환: D (n + 1, n + 1). 계수 벡터 c(상수항부터)에 대해 D @ c는 도함수의 계수 벡터
    """
    D = np.zeros((n + 1, n + 1))   # (n+1, n+1)
    for j in range(1, n + 1):
        D[j - 1, j] = j            # d/dx x^j = j x^(j-1)
    return D
