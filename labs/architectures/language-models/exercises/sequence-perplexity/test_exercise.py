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


perplexity = load().perplexity


@pytest.mark.parametrize("V", [2, 4, 50])
def test_uniform_logits_give_vocab_size(V):
    rng = np.random.default_rng(V)
    logits = np.full((7, V), 0.3)
    targets = rng.integers(V, size=7)
    assert np.isclose(perplexity(logits, targets), V), f"균등한 logit의 perplexity는 사전 크기 {V}여야 합니다"


def test_confident_correct_is_near_one():
    targets = np.array([0, 2, 1, 3])
    logits = np.full((4, 4), -20.0)
    logits[np.arange(4), targets] = 20.0
    assert abs(perplexity(logits, targets) - 1.0) < 1e-6, "정답에 거의 확률 1을 주면 perplexity는 1에 가까워야 합니다"


def test_hand_computed():
    # 정답 확률 0.5, 0.25: logit을 로그 확률로 직접 준다
    logits = np.log(np.array([[0.5, 0.5], [0.75, 0.25]]))
    targets = np.array([0, 1])
    expected = np.exp(-(np.log(0.5) + np.log(0.25)) / 2)      # = sqrt(8)
    assert np.isclose(perplexity(logits, targets), expected), "정답 확률의 기하평균의 역수와 같아야 합니다"


def test_large_logits_are_stable():
    logits = np.array([[1000.0, 999.0, 998.0], [1000.0, 1000.0, 0.0]])
    targets = np.array([0, 1])
    with np.errstate(over="raise", invalid="raise"):
        ppl = perplexity(logits, targets)
    p0 = 1 / (1 + np.exp(-1) + np.exp(-2))
    expected = np.exp(-(np.log(p0) + np.log(0.5)) / 2)
    assert np.isfinite(ppl) and np.isclose(ppl, expected), "큰 logit에서도 overflow 없이 계산해야 합니다(log-sum-exp)"


def test_shift_invariance():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(10, 6))
    targets = rng.integers(6, size=10)
    assert np.isclose(perplexity(logits, targets), perplexity(logits + 37.0, targets)), \
        "모든 logit에 같은 상수를 더해도 결과가 같아야 합니다"


def test_returns_float():
    out = perplexity(np.zeros((3, 5)), np.array([0, 1, 2]))
    assert np.ndim(out) == 0, "float 하나를 반환해야 합니다"
