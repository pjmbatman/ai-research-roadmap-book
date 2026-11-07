import numpy as np


def sbc_ranks(prior_sample, simulate, posterior_sample, n_sims, L, rng):
    """SBC 순위 배열 (n_sims,): 사후 표본 L개 중 참값보다 작은 것의 개수"""
    raise NotImplementedError("사전 → 시뮬레이션 → 사후 표본 → 순위를 n_sims번 반복하세요")


def rank_counts(ranks, L, n_bins):
    """순위 0..L을 같은 너비의 n_bins칸으로 센 개수 (n_bins,)"""
    raise NotImplementedError
