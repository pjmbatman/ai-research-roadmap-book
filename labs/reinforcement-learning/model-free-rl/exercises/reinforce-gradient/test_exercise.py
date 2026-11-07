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


reinforce_grad = load().reinforce_grad


def softmax(x):
    z = np.exp(x - x.max())
    return z / z.sum()


def test_single_sample_by_hand():
    theta = np.zeros(2)                      # pi = [0.5, 0.5]
    g = reinforce_grad(theta, np.array([0]), np.array([2.0]))
    assert np.shape(g) == (2,), f"기울기의 shape은 (A,) = (2,)여야 합니다. 현재 {np.shape(g)}"
    assert np.allclose(g, [1.0, -1.0]), "G * (onehot(a) - pi) = 2 * ([1, 0] - [0.5, 0.5]) = [1, -1]입니다"


def test_baseline_subtracted():
    theta = np.zeros(2)
    g = reinforce_grad(theta, np.array([0]), np.array([2.0]), baseline=2.0)
    assert np.allclose(g, 0.0), "G - b = 0이면 기울기 추정도 0입니다"


def test_components_sum_to_zero():
    rng = np.random.default_rng(0)
    theta = rng.normal(size=4)
    a = rng.integers(4, size=50)
    G = rng.normal(size=50)
    g = reinforce_grad(theta, a, G, baseline=0.3)
    assert np.isclose(np.sum(g), 0.0), "softmax 로짓에 대한 score onehot(a) - pi의 성분 합은 0입니다"


@pytest.mark.parametrize("baseline", [0.0, 1.0])
def test_unbiased_for_exact_gradient(baseline):
    rng = np.random.default_rng(1)
    theta = np.array([0.2, -0.4, 0.1])
    pi = softmax(theta)
    mu = np.array([1.0, 0.0, 0.5])           # 행동별 평균 보상
    a = rng.choice(3, size=200_000, p=pi)
    G = mu[a] + rng.normal(0, 0.1, size=a.size)
    exact = pi * (mu - pi @ mu)              # d/dtheta sum_a pi(a) mu(a)
    g = reinforce_grad(theta, a, G, baseline=baseline)
    assert np.allclose(g, exact, atol=5e-3), "표본이 많으면 추정이 정확한 기울기에 가까워야 합니다"


def test_large_logits_are_stable():
    theta = np.array([1000.0, 0.0])          # 큰 로짓에서도 softmax가 overflow하지 않아야 한다
    g = reinforce_grad(theta, np.array([1]), np.array([1.0]))
    assert np.all(np.isfinite(g)), "softmax를 계산할 때 최댓값을 빼서 overflow를 피하세요"
    assert np.allclose(g, [-1.0, 1.0], atol=1e-9)
