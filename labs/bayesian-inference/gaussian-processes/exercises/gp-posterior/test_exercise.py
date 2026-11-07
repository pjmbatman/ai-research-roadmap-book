import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()
X = np.array([-2.0, -0.5, 0.3, 1.0, 2.5])
Y = np.sin(X)


def test_shapes():
    mean, var = m.gp_posterior(X, Y, np.linspace(-3, 3, 7), 1.0, 1.0, 0.1)
    assert mean.shape == (7,) and var.shape == (7,)


def test_interpolates_without_noise():
    mean, var = m.gp_posterior(X, Y, X, 1.0, 1.0, 1e-4)
    assert np.allclose(mean, Y, atol=1e-3), "잡음이 거의 없으면 사후 평균이 학습 점을 지나야 합니다"
    assert np.all(var < 1e-6), "잡음이 거의 없으면 학습 점에서 분산이 0에 가까워야 합니다"


def test_reverts_to_prior_far_away():
    mean, var = m.gp_posterior(X, Y, [50.0], 1.0, 1.5, 0.1)
    assert mean[0] == pytest.approx(0.0, abs=1e-8), "멀리서는 사전 평균 0으로 돌아가야 합니다"
    assert var[0] == pytest.approx(1.5**2), "멀리서는 사전분산 sf^2로 돌아가야 합니다"


def test_single_point_formula():
    mean, var = m.gp_posterior([0.0], [2.0], [0.0], 1.0, 1.0, 1.0)
    assert mean[0] == pytest.approx(1.0) and var[0] == pytest.approx(0.5)


def test_lml_matches_mvn():
    ell, sf, sn = 0.8, 1.3, 0.2
    K = sf**2 * np.exp(-0.5 * (X[:, None] - X[None, :]) ** 2 / ell**2) + sn**2 * np.eye(5)
    expected = multivariate_normal(np.zeros(5), K).logpdf(Y)
    assert m.log_marginal_likelihood(list(X), list(Y), ell, sf, sn) == pytest.approx(expected)


def test_lml_prefers_reasonable_lengthscale():
    x = np.linspace(0, 10, 40)
    y = np.sin(x)
    lml = {ell: m.log_marginal_likelihood(x, y, ell, 1.0, 0.05) for ell in (0.05, 1.0, 20.0)}
    assert lml[1.0] > lml[0.05] and lml[1.0] > lml[20.0], "주변우도는 너무 짧거나 너무 긴 길이척도를 벌해야 합니다"
