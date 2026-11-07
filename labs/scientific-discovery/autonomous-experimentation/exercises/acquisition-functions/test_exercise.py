import importlib.util
import os
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load()
ucb, expected_improvement = mod.ucb, mod.expected_improvement


def test_ucb_values():
    out = ucb(np.array([0.5, 0.2]), np.array([0.1, 0.3]), beta=2.0)
    assert np.allclose(out, [0.7, 0.8]), f"UCB = mu + beta*sd입니다. 현재 {out}"


def test_ei_hand_value():
    ei = expected_improvement(np.array([1.0]), np.array([1.0]), best=1.0)   # z = 0
    assert np.allclose(ei, norm.pdf(0)), f"mu = best이면 EI = sd * φ(0) ≈ 0.399입니다. 현재 {ei}"


def test_ei_matches_monte_carlo():
    rng = np.random.default_rng(0)
    mu, sd, best = np.array([0.3, 0.6, 0.9]), np.array([0.2, 0.1, 0.4]), 0.65
    Y = mu + sd * rng.normal(size=(400000, 3))
    mc = np.maximum(Y - best, 0).mean(0)
    assert np.allclose(expected_improvement(mu, sd, best), mc, atol=2e-3), "몬테카를로 추정과 다릅니다. 최대화 문제의 부호를 확인하세요"


def test_zero_sd():
    ei = expected_improvement(np.array([0.9, 0.4]), np.array([0.0, 0.0]), best=0.5)
    assert np.allclose(ei, [0.4, 0.0]), f"sd = 0이면 EI = max(mu - best, 0)입니다. 현재 {ei}"
    assert np.all(np.isfinite(ei)), "sd = 0에서 nan이 나오면 안 됩니다"


def test_uncertainty_raises_ei():
    ei = expected_improvement(np.array([0.5, 0.5]), np.array([0.05, 0.3]), best=0.6)
    assert ei[1] > ei[0] > 0, "평균이 같으면 불확실성이 큰 쪽의 EI가 커야 합니다(탐색)"
    assert np.all(expected_improvement(np.linspace(-1, 1, 50), np.full(50, 0.1), 2.0) >= 0), "EI는 음수가 될 수 없습니다"
