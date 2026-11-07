import numpy as np


def identity_clusters(seqs, threshold):
    """길이가 같은 서열들의 단일 연결 군집 번호.

    동일성(같은 위치에 같은 글자의 비율) >= threshold인 쌍을 잇는 그래프의 연결 성분.
    번호는 처음 등장하는 순서대로 0, 1, 2, ...
    반환: (len(seqs),) 정수 배열
    """
    raise NotImplementedError("동일성 행렬과 연결 성분을 구현하세요")
