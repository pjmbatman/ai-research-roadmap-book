import numpy as np


def is_estimate(rewards, pb, pe, gamma=1.0):
    """기록된 궤적으로 평가 정책의 가치를 importance sampling으로 추정한다.

    rewards: (N, H) 궤적별·시각별 보상, pb: (N, H) 기록된 행동의 행동 정책 확률,
    pe: (N, H) 같은 행동의 평가 정책 확률, gamma: 할인율
    반환: (OIS 추정, WIS 추정, ESS)
    """
    raise NotImplementedError("궤적별 할인 이득 G와 가중치 w = prod(pe / pb)를 구해 세 값을 계산하세요")
