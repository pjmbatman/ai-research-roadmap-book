import numpy as np


def ucb_select(counts, sums, t, c=np.sqrt(2)):
    """UCB 규칙으로 다음에 당길 팔을 고른다.

    counts: (K,) 팔별 시행 수, sums: (K,) 팔별 보상 합, t: 지금까지의 총 시행 수(1 이상),
    c: 보너스 계수 (UCB1은 sqrt(2))
    반환: 팔 번호(int). 한 번도 당기지 않은 팔이 있으면 그중 번호가 가장 작은 팔.
    """
    raise NotImplementedError("평균 + c * sqrt(ln t / n)이 가장 큰 팔을 고르세요")
