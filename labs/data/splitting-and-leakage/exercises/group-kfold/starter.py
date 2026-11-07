import numpy as np


def group_kfold(groups, n_splits):
    """한 그룹이 한 fold에만 들어가도록 샘플마다 fold 번호를 정한다.

    groups: (N,) 그룹 이름 (정수 또는 문자열)
    n_splits: fold 수 K
    반환: (N,) int 배열, 값은 0..K-1
    """
    raise NotImplementedError("그룹을 큰 것부터 가장 작은 fold에 넣으세요")
