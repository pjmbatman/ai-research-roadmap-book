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


def random_spd(d, seed):
    B = np.random.default_rng(seed).normal(size=(d, d))
    return B @ B.T + d * np.eye(d)


def test_bivariate_formula():
    s1, s2, rho = 1.0, 1.5, 0.8
    mu = np.array([1.0, -1.0])
    Sigma = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    mean, cov = m.condition(mu, Sigma, [0], [2.2])
    assert mean.shape == (1,) and cov.shape == (1, 1)
    assert mean[0] == pytest.approx(-1.0 + rho * s2 / s1 * (2.2 - 1.0))
    assert cov[0, 0] == pytest.approx(s2**2 * (1 - rho**2))


def test_matches_precision_form():
    Sigma, mu = random_spd(5, 0), np.arange(5.0)
    Lam = np.linalg.inv(Sigma)
    a, b = [0, 2, 4], [1, 3]
    x_b = np.array([0.5, -2.0])
    mean, cov = m.condition(mu, Sigma, b, x_b)
    L_aa, L_ab = Lam[np.ix_(a, a)], Lam[np.ix_(a, b)]
    assert np.allclose(cov, np.linalg.inv(L_aa)), "조건부 공분산은 정밀도 행렬 블록 Λ_aa의 역행렬과 같아야 합니다"
    assert np.allclose(mean, mu[a] - np.linalg.solve(L_aa, L_ab @ (x_b - mu[b])))


def test_covariance_does_not_depend_on_observation():
    Sigma, mu = random_spd(4, 1), np.zeros(4)
    _, c1 = m.condition(mu, Sigma, [2, 3], [0.0, 0.0])
    _, c2 = m.condition(mu, Sigma, [2, 3], [10.0, -7.0])
    assert np.allclose(c1, c2)
    assert np.allclose(c1, c1.T) and np.all(np.linalg.eigvalsh(c1) > 0)


def test_unsorted_observed_indices():
    Sigma, mu = random_spd(4, 2), np.array([1.0, 2.0, 3.0, 4.0])
    m1, c1 = m.condition(mu, Sigma, [3, 0], [0.1, -0.2])
    m2, c2 = m.condition(mu, Sigma, [0, 3], [-0.2, 0.1])
    assert np.allclose(m1, m2) and np.allclose(c1, c2), "obs_idx와 x_obs는 같은 순서로 짝지어집니다"
