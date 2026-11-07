import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rejection_sample = load().rejection_sample


def beta25(x):
    return 30 * x * (1 - x) ** 4               # Beta(2, 5), 최댓값 2.4576


def test_exact_count_and_range():
    s, _ = rejection_sample(beta25, 2.5, 1234, np.random.default_rng(0))
    assert np.shape(s) == (1234,), f"표본은 정확히 n개여야 합니다. 현재 shape {np.shape(s)}"
    assert np.all((s >= 0) & (s <= 1)), "표본은 [0, 1] 안에 있어야 합니다"


def test_moments():
    s, _ = rejection_sample(beta25, 2.5, 50_000, np.random.default_rng(1))
    assert abs(s.mean() - 2 / 7) < 0.005, f"Beta(2,5)의 평균은 2/7 ≈ 0.2857입니다. 현재 {s.mean():.4f}"
    assert abs(s.var() - 10 / 392) < 0.001, f"Beta(2,5)의 분산은 10/392 ≈ 0.0255입니다. 현재 {s.var():.4f}"


def test_distribution_shape():
    s, _ = rejection_sample(beta25, 2.5, 20_000, np.random.default_rng(2))
    assert stats.kstest(s, stats.beta(2, 5).cdf).pvalue > 0.001, "표본의 분포가 Beta(2,5)와 다릅니다"


@pytest.mark.parametrize("M", [2.5, 5.0])
def test_acceptance_rate(M):
    _, rate = rejection_sample(beta25, M, 20_000, np.random.default_rng(3))
    assert abs(rate - 1 / M) < 0.02, f"정규화된 밀도에서 수용률은 1/M = {1 / M:.3f}에 가까워야 합니다. 현재 {rate:.3f}"


def test_unnormalized_density():
    s, rate = rejection_sample(lambda x: 3 * beta25(x), 7.5, 20_000, np.random.default_rng(4))
    assert abs(s.mean() - 2 / 7) < 0.01, "정규화 상수가 달라도 같은 분포가 나와야 합니다"
    assert abs(rate - 3 / 7.5) < 0.02, "정규화되지 않은 밀도에서 수용률은 Z/M입니다"
