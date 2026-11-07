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


mod = load()
cross_entropy, kl_divergence = mod.cross_entropy, mod.kl_divergence


def random_dist(rng, k):
    w = rng.random(k) + 0.05
    return w / w.sum()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_decomposition(seed):
    rng = np.random.default_rng(seed)
    p, q = random_dist(rng, 6), random_dist(rng, 6)
    H = cross_entropy(p, p)
    assert np.isclose(cross_entropy(p, q), H + kl_divergence(p, q)), "H(p,q) = H(p) + KL(p||q)가 성립해야 합니다"
    assert kl_divergence(p, q) > 0, "서로 다른 분포의 KL은 양수여야 합니다"
    assert np.isclose(kl_divergence(p, p), 0.0), "KL(p||p) = 0이어야 합니다"


def test_known_value():
    p, q = np.array([0.5, 0.5]), np.array([0.25, 0.75])
    expected = 0.5 * np.log(2) + 0.5 * np.log(2 / 3)
    assert np.isclose(kl_divergence(p, q), expected), "자연로그(nat)로 계산했는지 확인하세요"


def test_asymmetry():
    p, q = np.array([0.7, 0.2, 0.1]), np.array([0.4, 0.4, 0.2])
    assert not np.isclose(kl_divergence(p, q), kl_divergence(q, p)), "KL은 대칭이 아닙니다"


def test_zero_in_p_is_ignored():
    p, q = np.array([0.5, 0.5, 0.0]), np.array([0.4, 0.6, 0.0])
    v = kl_divergence(p, q)
    assert np.isfinite(v), "p_k = 0인 항은 q_k와 상관없이 0이어야 합니다 (nan이나 inf가 나오면 안 됩니다)"
    assert np.isclose(v, 0.5 * np.log(0.5 / 0.4) + 0.5 * np.log(0.5 / 0.6))
    assert np.isfinite(cross_entropy(p, q))


def test_support_mismatch_is_inf():
    p, q = np.array([0.7, 0.2, 0.1]), np.array([0.5, 0.5, 0.0])
    assert kl_divergence(p, q) == np.inf, "p > 0인데 q = 0인 곳이 있으면 inf여야 합니다"
    assert cross_entropy(p, q) == np.inf
    assert np.isfinite(kl_divergence(q, p)), "반대 방향 KL(q||p)는 유한합니다"
