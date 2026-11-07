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


infonce = load().infonce


def test_uninformative_critic_is_zero():
    assert abs(infonce(np.full((8, 8), 3.7))) < 1e-12, "모든 점수가 같으면 InfoNCE는 0이어야 합니다"


@pytest.mark.parametrize("K", [4, 32, 128])
def test_never_exceeds_log_k(K):
    rng = np.random.default_rng(K)
    for _ in range(20):
        S = rng.normal(size=(K, K)) * rng.uniform(0.1, 20)
        S[np.arange(K), np.arange(K)] += rng.uniform(0, 30)
        assert infonce(S) <= np.log(K) + 1e-9, "InfoNCE는 log K를 넘을 수 없습니다"


def test_approaches_log_k():
    K = 64
    S = np.zeros((K, K))
    S[np.arange(K), np.arange(K)] = 50.0
    assert np.isclose(infonce(S), np.log(K), atol=1e-6), "대각선이 압도적이면 log K에 다가가야 합니다"


def test_large_scores_are_stable():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(16, 16)) * 1000
    v = infonce(S)
    assert np.isfinite(v), "점수가 커도 overflow 없이 계산되어야 합니다 (log-sum-exp)"


def test_row_shift_invariance():
    rng = np.random.default_rng(1)
    S = rng.normal(size=(10, 10))
    shift = rng.normal(size=(10, 1)) * 5
    assert np.isclose(infonce(S), infonce(S + shift)), "각 행에 상수를 더해도 값이 같아야 합니다"


def test_relation_to_cross_entropy():
    rng = np.random.default_rng(2)
    K = 12
    S = rng.normal(size=(K, K)) * 3
    logp = S - np.log(np.sum(np.exp(S), axis=1, keepdims=True))
    ce = -np.mean(np.diag(logp))                             # 행마다 진짜 짝을 정답으로 한 cross-entropy
    assert np.isclose(infonce(S), np.log(K) - ce), "InfoNCE = log K - (contrastive cross-entropy)여야 합니다"
