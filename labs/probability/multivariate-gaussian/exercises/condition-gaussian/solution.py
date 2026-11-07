import numpy as np


def condition(mu, Sigma, obs_idx, x_obs):
    """x ~ N(mu, Sigma)에서 obs_idx 성분이 x_obs일 때 나머지 성분(인덱스 오름차순)의 조건부 평균·공분산.

    반환: (cond_mean (D-k,), cond_cov (D-k, D-k))
    """
    mu, Sigma = np.asarray(mu, dtype=float), np.asarray(Sigma, dtype=float)
    b = np.asarray(obs_idx, dtype=int)                       # (k,) 관측
    a = np.setdiff1d(np.arange(mu.size), b)                  # (D-k,) 미관측, 오름차순
    S_ab, S_bb = Sigma[np.ix_(a, b)], Sigma[np.ix_(b, b)]
    gain = np.linalg.solve(S_bb, S_ab.T).T                   # (D-k, k) = S_ab S_bb^{-1}
    cond_mean = mu[a] + gain @ (np.asarray(x_obs, dtype=float) - mu[b])
    cond_cov = Sigma[np.ix_(a, a)] - gain @ S_ab.T
    return cond_mean, 0.5 * (cond_cov + cond_cov.T)          # 반올림 오차로 깨진 대칭 복원
