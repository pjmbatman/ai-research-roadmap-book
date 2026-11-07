import numpy as np


def contact_map(X, cutoff=8.0, min_sep=6):
    """Cα 좌표 X (L, 3)에서 거리 < cutoff 이고 |i-j| >= min_sep 인 쌍의 (L, L) bool 행렬."""
    X = np.asarray(X, float)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)    # (L, L)
    idx = np.arange(len(X))
    sep = np.abs(idx[:, None] - idx[None, :])                     # (L, L)
    return (D < cutoff) & (sep >= min_sep)


def top_l_precision(scores, true_contacts, min_sep=6):
    """|i-j| >= min_sep 인 쌍 (i < j) 중 점수 상위 L개의 실제 접촉 비율."""
    scores = np.asarray(scores, float)
    L = scores.shape[0]
    i, j = np.triu_indices(L, k=min_sep)                          # j - i >= min_sep
    order = np.argsort(-scores[i, j])[:L]                         # 점수 상위 L개
    return float(np.mean(np.asarray(true_contacts)[i[order], j[order]]))
