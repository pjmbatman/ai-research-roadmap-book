import numpy as np


def permutation_importance(predict, X, y, n_repeats=5, seed=0):
    """특징마다 값을 섞었을 때 늘어나는 MSE의 평균을 반환한다.

    predict: (N, p) → (N,) 함수, X: (N, p), y: (N,)
    반환: (p,) 중요도
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    base = np.mean((predict(X) - y) ** 2)                # 기준 MSE
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        for _ in range(n_repeats):
            Xp = X.copy()                                # 원본은 그대로 둔다
            Xp[:, j] = rng.permutation(Xp[:, j])
            out[j] += np.mean((predict(Xp) - y) ** 2) - base
    return out / n_repeats
