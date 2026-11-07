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


selection_bias = load().selection_bias


def test_perfect_model_zero_bias():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    assert abs(selection_bias(y.copy(), y.copy(), 20)) < 1e-9, "예측 = 참이면 편향은 0이어야 합니다"


def test_picks_by_prediction_not_truth():
    # 참값은 모두 0, 예측은 무작위. top-k는 예측 큰 것 → pred>0, true=0 → 편향 = top-k 예측 평균 > 0
    rng = np.random.default_rng(1)
    pred = rng.normal(size=500)
    true = np.zeros(500)
    b = selection_bias(pred, true, 25)
    assert b > 0, "참값이 모두 0일 때, 예측 상위를 고르면 편향은 양수여야 합니다"
    assert np.isclose(b, np.sort(pred)[::-1][:25].mean()), "예측 상위 k개의 pred 평균과 같아야 합니다(true=0)"


def test_larger_than_random_selection():
    # 예측 = 참 + 잡음. 예측 기준 선별의 편향 > 무작위 선별의 편향(≈0)
    rng = np.random.default_rng(2)
    true = rng.normal(size=1000)
    pred = true + rng.normal(size=1000) * 1.0
    by_pred = selection_bias(pred, true, 50)
    rand_idx = rng.permutation(1000)[:50]
    by_random = np.mean(pred[rand_idx] - true[rand_idx])
    assert by_pred > by_random + 0.3, "예측 상위 선별의 낙관 편향이 무작위 선별보다 뚜렷이 커야 합니다"


def test_bias_grows_with_noise():
    rng = np.random.default_rng(3)
    true = rng.normal(size=1000)
    b_small = selection_bias(true + 0.2 * rng.normal(size=1000), true, 50)
    b_large = selection_bias(true + 1.5 * rng.normal(size=1000), true, 50)
    assert b_large > b_small, "예측 잡음이 클수록 낙관 편향이 커져야 합니다"
