import numpy as np


def scaffold_split(scaffolds, test_frac=0.2):
    """골격 단위로 train/test 인덱스를 나눈다.

    scaffolds: 길이 N의 골격 레이블 목록
    test_frac: test 비율의 목표값
    반환: (train_idx, test_idx), 각각 정렬된 정수 배열
    """
    groups = {}                                   # 골격 → 인덱스 목록 (처음 나타난 순서 유지)
    for i, s in enumerate(scaffolds):
        groups.setdefault(s, []).append(i)
    order = sorted(groups.values(), key=lambda idx: (-len(idx), idx[0]))
    n_train_max = (1 - test_frac) * len(scaffolds)
    train, test = [], []
    for idx in order:
        if len(train) + len(idx) <= n_train_max:
            train += idx
        else:
            test += idx
    return np.array(sorted(train), dtype=int), np.array(sorted(test), dtype=int)
