import numpy as np


def scaffold_split(scaffolds, test_frac=0.2):
    """골격 단위로 train/test 인덱스를 나눈다.

    scaffolds: 길이 N의 골격 레이블 목록
    test_frac: test 비율의 목표값
    반환: (train_idx, test_idx), 각각 정렬된 정수 배열
    """
    raise NotImplementedError("골격을 크기순으로 정렬해 train부터 채우세요")
