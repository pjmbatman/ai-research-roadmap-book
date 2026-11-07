import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chisquare

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()
A, B, N = 3.0, 0.5, 10                                   # lambda ~ Gamma(3, rate 0.5), y_i ~ Poisson(lambda)
prior = lambda rng: rng.gamma(A, 1 / B)
simulate = lambda lam, rng: rng.poisson(lam, size=N)


def posterior(var_scale=1.0, shift=0.0):
    def sample(y, L, rng):
        shape = (A + y.sum()) / var_scale                # var_scale < 1: 평균은 같고 분산만 줄인다
        mean = (A + y.sum()) / (B + N)
        return rng.gamma(shape, mean / shape, size=L) + shift
    return sample


def run(sample, n_sims=500, seed=0):
    ranks = m.sbc_ranks(prior, simulate, sample, n_sims, 99, np.random.default_rng(seed))
    return ranks, m.rank_counts(ranks, 99, 10)


def test_rank_counts():
    counts = m.rank_counts(np.array([0, 9, 10, 55, 99, 99]), 99, 10)
    assert list(counts) == [2, 1, 0, 0, 0, 1, 0, 0, 0, 2]


def test_ranks_shape_and_range():
    ranks, counts = run(posterior(), n_sims=50)
    assert ranks.shape == (50,) and ranks.min() >= 0 and ranks.max() <= 99
    assert counts.sum() == 50


def test_correct_posterior_is_uniform():
    _, counts = run(posterior())
    assert chisquare(counts).pvalue > 0.01, "올바른 사후분포의 순위는 균일해야 합니다"


def test_overconfident_posterior_is_u_shaped():
    _, counts = run(posterior(var_scale=0.25))
    assert chisquare(counts).pvalue < 1e-6
    ends, middle = counts[0] + counts[-1], counts[1:-1].mean()     # 균일하면 ends ≈ 2 * middle
    assert ends > 5 * middle, "과신한 사후분포는 양 끝 칸에 순위가 몰립니다"


def test_biased_posterior_is_skewed():
    _, counts = run(posterior(shift=2.0))
    assert counts[:5].sum() > 2 * counts[5:].sum(), "사후분포가 위로 치우치면 참값의 순위가 낮은 쪽에 몰립니다"
