import numpy as np


def derivative_matrix(n):
    """P_n 위의 미분 연산자를 단항식 basis (1, x, ..., x^n)로 표현한 행렬.

    n: 다항식의 최고 차수 (n >= 0)
    반환: D (n + 1, n + 1). 계수 벡터 c(상수항부터)에 대해 D @ c는 도함수의 계수 벡터
    """
    raise NotImplementedError("D의 각 열에 D(x^j)의 좌표를 채우세요")
