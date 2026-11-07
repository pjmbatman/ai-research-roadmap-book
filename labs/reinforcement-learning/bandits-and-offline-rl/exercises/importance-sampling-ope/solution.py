import numpy as np


def is_estimate(rewards, pb, pe, gamma=1.0):
    """기록된 궤적으로 평가 정책의 가치를 importance sampling으로 추정한다.

    rewards: (N, H) 궤적별·시각별 보상, pb: (N, H) 기록된 행동의 행동 정책 확률,
    pe: (N, H) 같은 행동의 평가 정책 확률, gamma: 할인율
    반환: (OIS 추정, WIS 추정, ESS)
    """
    rewards, pb, pe = (np.asarray(x, dtype=float) for x in (rewards, pb, pe))
    H = rewards.shape[1]
    G = rewards @ gamma ** np.arange(H)                  # (N,) 궤적별 할인 이득
    w = np.prod(pe / pb, axis=1)                         # (N,) 궤적별 importance weight
    ois = np.mean(w * G)
    wis = np.sum(w * G) / np.sum(w)
    ess = np.sum(w) ** 2 / np.sum(w**2)
    return float(ois), float(wis), float(ess)
