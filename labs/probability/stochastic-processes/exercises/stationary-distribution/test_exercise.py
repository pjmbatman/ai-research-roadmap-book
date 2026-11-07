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


stationary_distribution = load().stationary_distribution


@pytest.mark.parametrize("a,b", [(0.1, 0.3), (0.5, 0.5), (0.02, 0.9)])
def test_two_state_formula(a, b):
    P = np.array([[1 - a, a], [b, 1 - b]])
    pi = stationary_distribution(P)
    assert np.allclose(pi, [b / (a + b), a / (a + b)]), f"두 상태 체인의 정상분포는 (b, a)/(a+b)입니다. 현재 {pi}"


def test_doubly_stochastic_is_uniform():
    P = np.array([[0.2, 0.5, 0.3], [0.3, 0.2, 0.5], [0.5, 0.3, 0.2]])
    assert np.allclose(stationary_distribution(P), np.full(3, 1 / 3)), "열의 합도 1이면 정상분포는 균등분포입니다"


def test_not_right_eigenvector():
    P = np.array([[0.90, 0.08, 0.02], [0.10, 0.80, 0.10], [0.05, 0.15, 0.80]])
    pi = stationary_distribution(P)
    assert not np.allclose(pi, np.full(3, 1 / 3)), "P의 (오른쪽) 고유벡터를 쓰면 모든 성분이 같은 벡터가 나옵니다. P^T를 쓰세요"
    assert np.allclose(pi @ P, pi)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_chain(seed):
    rng = np.random.default_rng(seed)
    P = rng.uniform(size=(6, 6))
    P /= P.sum(axis=1, keepdims=True)
    pi = stationary_distribution(P)
    assert np.shape(pi) == (6,), f"shape은 (K,) = (6,)이어야 합니다. 현재 {np.shape(pi)}"
    assert np.isclose(np.sum(pi), 1.0), "정상분포의 합은 1이어야 합니다"
    assert np.all(np.asarray(pi) >= 0), "정상분포의 성분은 음수일 수 없습니다"
    assert np.allclose(pi @ P, pi), "pi P = pi를 만족해야 합니다"
    p = np.eye(6)[0]
    for _ in range(200):                                  # 멱반복으로 얻은 극한과도 같아야 한다
        p = p @ P
    assert np.allclose(pi, p, atol=1e-8)
