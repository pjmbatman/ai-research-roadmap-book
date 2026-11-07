import numpy as np


def invariant_features(X):
    """점구름 X (N, 2) → 회전·평행이동·순서 바꾸기에 불변인 1차원 특징 벡터.

    같은 N이면 같은 길이를 반환한다.
    """
    X = np.asarray(X, dtype=float)
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)   # (N, N) 쌍별 거리: 회전·평행이동 불변
    iu = np.triu_indices(len(X), k=1)
    return np.sort(d[iu])                                        # (N(N-1)/2,) 정렬: 순서 불변
