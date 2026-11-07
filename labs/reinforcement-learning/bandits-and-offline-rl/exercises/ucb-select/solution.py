import numpy as np


def ucb_select(counts, sums, t, c=np.sqrt(2)):
    """UCB 규칙으로 다음에 당길 팔을 고른다.

    counts: (K,) 팔별 시행 수, sums: (K,) 팔별 보상 합, t: 지금까지의 총 시행 수(1 이상),
    c: 보너스 계수 (UCB1은 sqrt(2))
    반환: 팔 번호(int). 한 번도 당기지 않은 팔이 있으면 그중 번호가 가장 작은 팔.
    """
    counts = np.asarray(counts, dtype=float)
    sums = np.asarray(sums, dtype=float)
    untried = np.flatnonzero(counts == 0)
    if untried.size:
        return int(untried[0])
    index = sums / counts + c * np.sqrt(np.log(t) / counts)    # (K,) 평균 + 불확실성 보너스
    return int(np.argmax(index))
