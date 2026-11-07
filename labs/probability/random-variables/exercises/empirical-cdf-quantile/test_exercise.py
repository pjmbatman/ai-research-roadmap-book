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


def test_ecdf_small_with_ties():
    s = [3.0, 1.0, 1.0, 2.0]
    got = m.ecdf(s, np.array([0.0, 1.0, 1.5, 2.0, 3.0, 10.0]))
    assert np.allclose(got, [0.0, 0.5, 0.5, 0.75, 1.0, 1.0]), "F_n(x)는 x '이하'인 표본의 비율입니다(동점 포함)"


def test_ecdf_shape():
    s = np.arange(10.0)
    assert np.shape(m.ecdf(s, np.zeros((2, 3)))) == (2, 3)


def test_quantile_small():
    s = [3.0, 1.0, 1.0, 2.0]
    got = m.quantile(s, np.array([0.1, 0.5, 0.51, 0.75, 1.0]))
    assert np.allclose(got, [1.0, 1.0, 2.0, 2.0, 3.0]), "Q_n(p)는 F_n이 처음으로 p 이상이 되는 표본값입니다"


def test_matches_numpy_inverted_cdf():
    rng = np.random.default_rng(0)
    s = rng.normal(size=37)
    p = np.linspace(0.01, 1.0, 100)
    assert np.allclose(m.quantile(s, p), np.quantile(s, p, method="inverted_cdf"))


def test_galois_relations():
    rng = np.random.default_rng(1)
    s = rng.poisson(3.0, size=50).astype(float)            # 동점이 많은 이산 표본
    p = np.linspace(0.02, 1.0, 50)
    assert np.all(m.ecdf(s, m.quantile(s, p)) >= p - 1e-12), "F_n(Q_n(p)) >= p 이어야 합니다"
    assert np.all(m.quantile(s, m.ecdf(s, s)) <= s + 1e-12), "Q_n(F_n(x)) <= x 이어야 합니다"


def test_large_sample_close_to_true_cdf():
    rng = np.random.default_rng(2)
    s = rng.exponential(scale=2.0, size=20000)
    x = np.array([0.5, 1.0, 2.0, 5.0])
    assert np.allclose(m.ecdf(s, x), 1 - np.exp(-x / 2.0), atol=0.015)
    assert m.quantile(s, 0.5) == pytest.approx(2.0 * np.log(2), abs=0.05)
