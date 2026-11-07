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


def test_small_values():
    eta = np.array([0.0, np.log(2.0), np.log(3.0)])
    assert m.log_partition(eta) == pytest.approx(np.log(6.0))
    assert np.allclose(m.mean_params(eta), [1 / 6, 2 / 6, 3 / 6])


def test_large_logits_are_stable():
    eta = np.array([1000.0, 1001.0, 999.0])
    A = m.log_partition(eta)
    assert np.isfinite(A), "logit이 크면 exp가 overflow합니다. 최댓값을 먼저 빼세요"
    assert A == pytest.approx(1001.0 + np.log(1 + np.exp(-1) + np.exp(-2)))
    pi = m.mean_params(eta)
    assert np.all(np.isfinite(pi)) and pi.sum() == pytest.approx(1.0)
    assert np.isfinite(m.log_prob(np.array([-1000.0, 1000.0]), 0)), "아주 작은 확률의 로그도 유한해야 합니다"
    assert m.log_prob(np.array([-1000.0, 1000.0]), 0) == pytest.approx(-2000.0)


def test_shift_invariance():
    rng = np.random.default_rng(0)
    eta = rng.normal(size=5)
    assert np.allclose(m.mean_params(eta), m.mean_params(eta + 37.0)), "모든 logit에 같은 상수를 더해도 확률은 같습니다"
    assert m.log_partition(eta + 37.0) == pytest.approx(m.log_partition(eta) + 37.0)


def test_gradient_of_log_partition_is_mean():
    rng = np.random.default_rng(1)
    eta, h = rng.normal(size=4), 1e-5
    grad = np.array([(m.log_partition(eta + h * e) - m.log_partition(eta - h * e)) / (2 * h) for e in np.eye(4)])
    assert np.allclose(grad, m.mean_params(eta), atol=1e-7), "dA/d(eta_k)는 pi_k여야 합니다"


def test_log_prob_consistent():
    eta = np.array([0.3, -1.2, 2.0])
    for k in range(3):
        assert m.log_prob(eta, k) == pytest.approx(np.log(m.mean_params(eta)[k]))
