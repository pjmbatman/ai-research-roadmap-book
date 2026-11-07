import numpy as np


def group_kfold(groups, n_splits):
    """한 그룹이 한 fold에만 들어가도록 샘플마다 fold 번호를 정한다.

    groups: (N,) 그룹 이름 (정수 또는 문자열)
    n_splits: fold 수 K
    반환: (N,) int 배열, 값은 0..K-1
    """
    groups = np.asarray(groups)
    names, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(names) < n_splits:
        raise ValueError(f"그룹 {len(names)}개로 {n_splits}개 fold를 만들 수 없습니다")
    order = sorted(range(len(names)), key=lambda g: (-counts[g], g))   # 큰 그룹부터, 같으면 이름 순
    fold_size = np.zeros(n_splits, dtype=int)                           # (K,)
    group_fold = np.empty(len(names), dtype=int)                        # (n_groups,)
    for g in order:
        k = int(np.argmin(fold_size))                                   # 지금 가장 작은 fold
        group_fold[g] = k
        fold_size[k] += counts[g]
    return group_fold[inverse.ravel()]
