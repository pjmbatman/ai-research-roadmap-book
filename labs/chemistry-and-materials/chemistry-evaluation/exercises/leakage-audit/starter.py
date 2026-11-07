import numpy as np


def leaked_fraction(X, train_idx, test_idx, threshold):
    """test 분자 중 train에 Tanimoto ≥ threshold인 근접 중복을 가진 비율을 반환한다.

    X: (N, L) 비트 지문, train_idx/test_idx: 인덱스 배열, threshold: 임계
    """
    raise NotImplementedError("test×train Tanimoto의 행별 최댓값이 threshold 이상인 비율을 구하세요")
