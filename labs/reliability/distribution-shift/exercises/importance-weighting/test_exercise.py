import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()


def test_equal_weights_is_plain_mean():
    assert m.weighted_risk([1.0, 2.0, 6.0], [3.0, 3.0, 3.0]) == pytest.approx(3.0), "가중치가 모두 같으면 보통 평균입니다"


def test_self_normalized():
    # (1×1 + 3×2) / (1 + 3) = 7/4
    assert m.weighted_risk([1.0, 2.0], [1.0, 3.0]) == pytest.approx(1.75), "가중치 합으로 나눕니다"


def test_effective_sample_size():
    assert m.effective_sample_size(np.ones(10)) == pytest.approx(10.0)
    assert m.effective_sample_size([2.0, 2.0, 0.0, 0.0]) == pytest.approx(2.0)
    assert m.effective_sample_size([5.0, 0.0, 0.0]) == pytest.approx(1.0), "가중치가 한 점에 몰리면 유효 표본은 1입니다"


def test_weighted_line_fit_ignores_zero_weight_points():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 5.0, 100.0])                       # 마지막 점은 이상치
    a, b = m.weighted_line_fit(x, y, [1.0, 1.0, 1.0, 0.0])
    assert (a, b) == pytest.approx((1.0, 2.0)), "가중치 0인 점은 적합에 영향을 주지 않아야 합니다"


def test_estimates_target_risk_under_covariate_shift():
    rng = np.random.default_rng(0)
    model = lambda x: 1.0 + 0.5 * x                            # 고정된(틀린) 모델
    truth = lambda x: np.sin(x) + x
    x_s = rng.normal(0, 1, 4000)
    loss_s = (truth(x_s) + rng.normal(0, 0.2, x_s.size) - model(x_s)) ** 2
    w = norm.pdf(x_s, 0.8, 0.7) / norm.pdf(x_s, 0, 1)
    x_t = rng.normal(0.8, 0.7, 200000)
    actual = np.mean((truth(x_t) + rng.normal(0, 0.2, x_t.size) - model(x_t)) ** 2)
    est = m.weighted_risk(loss_s, w)
    assert abs(est - actual) < 0.05 * actual, f"가중 추정 {est:.3f}이 운영 손실 {actual:.3f}에 가까워야 합니다"
    assert abs(np.mean(loss_s) - actual) > 0.1 * actual        # 가중하지 않으면 틀린다
