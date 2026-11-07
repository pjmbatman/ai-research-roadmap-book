import numpy as np
from scipy.special import logsumexp


def eig_nmc(mean_fn, theta_outer, theta_inner, sigma, rng):
    """가우시안 측정 잡음 모델에서 기대 정보 이득(EIG)의 중첩 몬테카를로 추정 [nat].

    mean_fn: (K, p) 파라미터 -> (K, n) 측정 평균
    theta_outer: (N, p) 바깥 사전 표본
    theta_inner: (M, p) 안쪽 사전 표본
    sigma: 측정 잡음 표준편차
    rng: np.random.Generator (가상 측정 잡음용, normal을 한 번 호출)
    반환: EIG 추정값 (실수)
    """
    mu_n = np.asarray(mean_fn(theta_outer), dtype=float)            # (N, n)
    mu_m = np.asarray(mean_fn(theta_inner), dtype=float)            # (M, n)
    y = mu_n + sigma * rng.normal(size=mu_n.shape)                  # (N, n) 가상 측정
    outer = -0.5 * (((y - mu_n) / sigma) ** 2).sum(-1)              # (N,)
    cross = -0.5 * (((y[:, None, :] - mu_m[None]) / sigma) ** 2).sum(-1)   # (N, M)
    inner = logsumexp(cross, axis=1) - np.log(len(mu_m))            # (N,)
    return float(np.mean(outer - inner))
