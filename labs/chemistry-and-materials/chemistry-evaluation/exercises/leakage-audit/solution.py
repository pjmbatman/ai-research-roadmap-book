import numpy as np


def leaked_fraction(X, train_idx, test_idx, threshold):
    """test 분자 중 train에 Tanimoto ≥ threshold인 근접 중복을 가진 비율을 반환한다.

    X: (N, L) 비트 지문, train_idx/test_idx: 인덱스 배열, threshold: 임계
    """
    A = np.asarray(X)[np.asarray(test_idx)].astype(float)    # (n_te, L)
    B = np.asarray(X)[np.asarray(train_idx)].astype(float)   # (n_tr, L)
    inter = A @ B.T                                           # (n_te, n_tr)
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    sim = inter / np.where(union > 0, union, 1.0)             # 둘 다 0비트면 유사도 0
    max_sim = sim.max(1)                                      # 각 test 분자의 최근접 train 유사도
    return float(np.mean(max_sim >= threshold))
