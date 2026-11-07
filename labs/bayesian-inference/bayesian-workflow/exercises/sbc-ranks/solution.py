import numpy as np


def sbc_ranks(prior_sample, simulate, posterior_sample, n_sims, L, rng):
    """SBC 순위 배열 (n_sims,): 사후 표본 L개 중 참값보다 작은 것의 개수"""
    ranks = np.empty(n_sims, dtype=int)
    for s in range(n_sims):
        theta = prior_sample(rng)                       # 1) 사전에서 참값
        data = simulate(theta, rng)                     # 2) 데이터 시뮬레이션
        draws = np.asarray(posterior_sample(data, L, rng))   # 3) (L,) 사후 표본
        ranks[s] = int(np.sum(draws < theta))           # 4) 순위
    return ranks


def rank_counts(ranks, L, n_bins):
    """순위 0..L을 같은 너비의 n_bins칸으로 센 개수 (n_bins,)"""
    width = (L + 1) // n_bins
    return np.bincount(np.asarray(ranks) // width, minlength=n_bins)
