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


fit_predict_proba = load().fit_predict_proba


def test_hand_computed_example():
    # 클래스 0: 0, 2 (평균 1), 클래스 1: 4, 6 (평균 5), 공유 분산 = 1, 사전확률 1/2씩
    x, y = np.array([0.0, 2.0, 4.0, 6.0]), np.array([0, 0, 1, 1])
    p = np.asarray(fit_predict_proba(x, y, np.array([3.0, 4.0])))
    assert np.shape(p) == (2,), f"반환 shape은 (M,) = (2,)여야 합니다. 현재 {np.shape(p)}"
    assert np.isclose(p[0], 0.5), f"두 평균의 한가운데(x=3)에서는 0.5입니다. 현재 {p[0]}"
    expected = 1 / (1 + np.exp(-4.0))                         # logit = ((4-1)^2 - (4-5)^2) / 2 = 4
    assert np.isclose(p[1], expected), f"x=4에서 p(y=1|x) = {expected:.4f}입니다. 현재 {p[1]:.4f}"


def test_prior_shifts_probability():
    # 클래스 1이 3배 많으면 한가운데에서 p(y=1|x) = 3/4
    x = np.array([0.0, 2.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0])
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    p = np.asarray(fit_predict_proba(x, y, np.array([3.0])))
    assert np.isclose(p[0], 0.75), f"사전확률 3:1이면 평균 한가운데에서 0.75입니다. 현재 {p[0]}"


def test_probabilities_in_range_far_from_data():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 50), rng.normal(3, 1, 50)])
    y = np.repeat([0, 1], 50)
    p = np.asarray(fit_predict_proba(x, y, np.array([-200.0, 0.0, 1.5, 3.0, 200.0])))
    assert np.all(np.isfinite(p)), "데이터에서 먼 점에서도 NaN이 나오면 안 됩니다(로그 밀도로 계산)"
    assert np.all((p >= 0) & (p <= 1))
    assert p[0] < 1e-6 and p[-1] > 1 - 1e-6
    assert np.all(np.diff(p) > -1e-12), "평균이 큰 클래스 쪽으로 갈수록 확률이 커져야 합니다"


def test_logit_is_linear_in_x():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(-1, 0.7, 80), rng.normal(1.5, 0.7, 40)])
    y = np.repeat([0, 1], [80, 40])
    grid = np.linspace(-1, 1.5, 11)                           # 확률이 0이나 1에 붙지 않는 구간
    p = np.asarray(fit_predict_proba(x, y, grid))
    logit = np.log(p) - np.log1p(-p)
    second_diff = np.diff(logit, 2)
    assert np.allclose(second_diff, 0, atol=1e-6), "분산을 공유하면 log(p/(1-p))는 x의 1차 함수여야 합니다"


def test_matches_reference_formula():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(0, 2, 30), rng.normal(1, 2, 70)])
    y = np.repeat([0, 1], [30, 70])
    mu0, mu1 = x[y == 0].mean(), x[y == 1].mean()
    var = np.mean((x - np.where(y == 1, mu1, mu0)) ** 2)
    x_new = np.array([-1.0, 0.5, 2.0])
    n0 = 0.3 * np.exp(-(x_new - mu0) ** 2 / (2 * var))
    n1 = 0.7 * np.exp(-(x_new - mu1) ** 2 / (2 * var))
    assert np.allclose(fit_predict_proba(x, y, x_new), n1 / (n0 + n1)), "공유 분산은 전체 N으로 나눈 MLE입니다"
