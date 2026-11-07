import numpy as np


def split_indices(n, fractions=(0.6, 0.2, 0.2), seed=0):
    """0..n-1을 seed로 섞어 학습·검증·테스트 인덱스로 나눈다.

    n_train = int(f_train * n), n_val = int(f_val * n), 나머지는 테스트.
    전역 난수 상태를 쓰지도 바꾸지도 않는다.
    반환: (train_idx, val_idx, test_idx) 정수 배열 세 개
    """
    raise NotImplementedError("자기만의 난수 생성기로 섞고 잘라 나누세요")
