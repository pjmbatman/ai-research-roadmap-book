import numpy as np


def permutation_importance(predict, X, y, cols, n_repeats=10, seed=0):
    """열 cols를 같은 순열로 함께 섞었을 때 평균 제곱오차가 늘어나는 양의 평균.

    predict: (N, D) -> (N,), X: (N, D), y: (N,), cols: 열 번호 리스트
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    base = np.mean((predict(X) - y) ** 2)
    increases = []
    for _ in range(n_repeats):
        Xp = X.copy()
        perm = rng.permutation(len(X))
        Xp[:, cols] = X[perm][:, cols]          # 묶음의 열들을 같은 순열로 섞는다
        increases.append(np.mean((predict(Xp) - y) ** 2) - base)
    return float(np.mean(increases))
