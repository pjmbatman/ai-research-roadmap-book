import numpy as np


def best_threshold(y, p, c_fn, c_fp):
    """총비용 c_fn * FN + c_fp * FP를 최소화하는 threshold를 고른다.

    y: (N,) 0 또는 1, p: (N,) 양성 점수. p >= t이면 양성으로 판정한다.
    반환: (t, cost). 후보는 np.unique(p)와 np.inf, 비용이 같으면 가장 작은 t.
    """
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    cand = np.append(np.unique(p), np.inf)                      # (M,) 오름차순
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    fn = np.searchsorted(pos, cand, side="left")                # p < t인 양성의 수, (M,)
    fp = len(neg) - np.searchsorted(neg, cand, side="left")     # p >= t인 음성의 수, (M,)
    cost = c_fn * fn + c_fp * fp
    j = int(np.argmin(cost))                                    # 첫 번째 최솟값 = 가장 작은 t
    return float(cand[j]), float(cost[j])
