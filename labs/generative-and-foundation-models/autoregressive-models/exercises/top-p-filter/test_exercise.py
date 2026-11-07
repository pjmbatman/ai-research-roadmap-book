import importlib.util
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sample_probs = load().sample_probs


def test_plain_softmax():
    p = sample_probs(np.log([0.5, 0.3, 0.2]))
    assert np.allclose(p, [0.5, 0.3, 0.2]), f"온도 1, top_p 1이면 softmax 그대로입니다. 현재 {p}"


def test_temperature_sharpens_and_flattens():
    lg = np.log([0.6, 0.3, 0.1])
    cold, hot = sample_probs(lg, temperature=0.5), sample_probs(lg, temperature=2.0)
    assert np.allclose(cold, np.array([0.36, 0.09, 0.01]) / 0.46), "T=0.5는 확률을 제곱한 뒤 정규화한 것과 같습니다"
    assert hot.max() < 0.6 < cold.max(), "T>1은 평평하게, T<1은 뾰족하게 만듭니다"


def test_large_logits_do_not_overflow():
    p = sample_probs(np.array([1000.0, 999.0, 0.0]))
    assert np.all(np.isfinite(p)) and np.isclose(p.sum(), 1.0), "최댓값을 빼고 exp를 취하세요"
    assert np.isclose(p[0] / p[1], np.e)


def test_nucleus_example_from_problem():
    p = sample_probs(np.log([0.5, 0.2, 0.15, 0.1, 0.05]), top_p=0.8)
    expect = np.array([0.5, 0.2, 0.15, 0.0, 0.0]) / 0.85
    assert np.allclose(p, expect), f"누적 0.85에서 처음 0.8 이상이 되므로 세 토큰이 남습니다. 현재 {np.round(p, 3)}"
    assert np.all(p[3:] == 0.0), "버린 토큰의 확률은 정확히 0이어야 합니다"


def test_unsorted_input():
    p = sample_probs(np.log([0.05, 0.15, 0.5, 0.1, 0.2]), top_p=0.8)
    assert np.allclose(p, np.array([0.0, 0.15, 0.5, 0.0, 0.2]) / 0.85), "입력 순서와 무관하게 확률이 큰 토큰부터 남깁니다"


def test_single_dominant_token():
    p = sample_probs(np.log([0.9, 0.05, 0.05]), top_p=0.5)
    assert np.allclose(p, [1.0, 0.0, 0.0]), "가장 큰 확률이 이미 top_p 이상이면 그 토큰 하나만 남습니다"


def test_temperature_applied_before_top_p():
    lg = np.log([0.5, 0.3, 0.2])
    p = sample_probs(lg, temperature=0.5, top_p=0.7)        # T=0.5: [25, 9, 4]/38 -> 0.658, 0.237, ...
    assert np.allclose(p, np.array([25, 9, 0]) / 34), f"온도를 먼저 적용한 분포에 top-p를 겁니다. 현재 {np.round(p, 3)}"
