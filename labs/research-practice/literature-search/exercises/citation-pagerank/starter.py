import numpy as np


def pagerank(A, d=0.85, tol=1e-12, max_iter=1000):
    """인용 그래프의 PageRank.

    A: (n, n) 배열. A[j, i] = 1이면 논문 i가 논문 j를 인용한다.
    d: damping factor. 참고문헌이 없는 논문의 열은 균등분포로 채운다.
    반환: (n,) 배열, 합은 1.
    """
    raise NotImplementedError("전이행렬을 만들고 power iteration을 구현하세요")
