import numpy as np


def pi_exponents(D, rep, target):
    """반복 변수 rep로 변수 target을 무차원화하는 π 그룹의 지수 벡터.

    D: (k, n) 차원 행렬. 열 i는 물리량 i의 [기본 차원별 지수]
    rep: 반복 변수의 열 번호 목록 (길이 k, D[:, rep]은 가역)
    target: 지수 1로 들어갈 변수의 열 번호
    반환: e (n,) — D @ e = 0, e[target] = 1, 나머지 비반복 변수는 0
    """
    raise NotImplementedError("D[:, rep] @ x = -D[:, target]을 풀어 지수 벡터를 채우세요")
