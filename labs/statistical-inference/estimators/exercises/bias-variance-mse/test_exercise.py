import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()


def normal(rng, size):
    return rng.normal(3.0, 2.0, size=size)   # 평균 3, 분산 4


def test_sample_mean_is_unbiased_with_variance_sigma2_over_n():
    bias, var, mse = m.bias_variance_mse(lambda X: X.mean(axis=1), normal, 3.0, n=10)
    assert abs(bias) < 0.03, "표본 평균은 불편 추정량입니다"
    assert var == pytest.approx(4.0 / 10, rel=0.05), "표본 평균의 분산은 σ²/n입니다"


def test_divide_by_n_variance_is_biased_downward():
    bias, var, mse = m.bias_variance_mse(lambda X: X.var(axis=1), normal, 4.0, n=5)
    assert bias == pytest.approx(-4.0 / 5, abs=0.05), "n으로 나눈 분산 추정량의 bias는 -σ²/n입니다"


def test_mse_decomposition_holds_exactly():
    bias, var, mse = m.bias_variance_mse(lambda X: X.var(axis=1, ddof=1), normal, 4.0, n=5, reps=5000)
    assert mse == pytest.approx(bias**2 + var, rel=1e-9), "variance를 reps로 나누면 MSE = bias² + variance가 정확히 성립합니다"


def test_same_seed_same_result():
    est = lambda X: np.median(X, axis=1)
    first = m.bias_variance_mse(est, normal, 3.0, n=7, reps=1000, seed=3)
    assert first == m.bias_variance_mse(est, normal, 3.0, n=7, reps=1000, seed=3), "같은 seed면 같은 결과가 나와야 합니다"


def test_median_is_less_efficient_than_mean_for_normal_data():
    _, var_mean, _ = m.bias_variance_mse(lambda X: X.mean(axis=1), normal, 3.0, n=50)
    _, var_med, _ = m.bias_variance_mse(lambda X: np.median(X, axis=1), normal, 3.0, n=50)
    assert 1.3 < var_med / var_mean < 1.8, "정규분포에서 중앙값의 분산은 평균의 약 π/2배입니다"
